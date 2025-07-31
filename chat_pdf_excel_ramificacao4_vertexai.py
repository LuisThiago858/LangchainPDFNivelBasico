"""
RAG para planilhas de indicadores financeiros
(com memória + filtro dinâmico por mês/ano + validação anti-alucinação
 + default para mês atual com fallback para último disponível + filtro por indicador
 + fallback de LLM para Gemini (Vertex) em caso de timeout do OpenAI
 + fallback local opcional com HuggingFace + Chroma
 + Vertex AI Vector Search como banco vetorial gerenciado no GCP)

Requisitos:
    pip install -U langchain langchain-openai langchain-google-vertexai langchain-chroma \
                  langchain-community google-cloud-aiplatform python-dotenv pandas
    # (opcional para fallback local)
    pip install -U sentence-transformers langchain-huggingface
    # defina OPENAI_API_KEY no seu ambiente (.env ou variável de ambiente)
"""

# ========= 0. Imports ========= #
from __future__ import annotations
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

import os, re, shutil
import pandas as pd
from dotenv import load_dotenv

from langchain_google_vertexai import VertexAIEmbeddings, VectorSearchVectorStore, ChatVertexAI
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

from google.cloud import aiplatform
# Namespaces para filtros do Vertex AI Vector Search
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

import httpx
from openai import APITimeoutError

load_dotenv()

# ========= 1. Config ========= #
PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT", "sistem-ag")
REGION     = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
BUCKET     = os.getenv("VERTEX_STAGING_BUCKET", "gs://bucket-vetores-gisa")

INDEX_DISPLAY_NAME    = os.getenv("VERTEX_INDEX_NAME", "indices-financeiros-gisa")
ENDPOINT_DISPLAY_NAME = os.getenv("VERTEX_ENDPOINT_NAME", "endpoint-financeiros-gisa")

# text-embedding-005 => 768 dims
VECTOR_DIMENSIONS = int(os.getenv("VERTEX_VECTOR_DIM", "768"))

BASES = {
    "4": "indicadores_financeiros_almater.xlsx",
    "5": "indicadores_base_2.xlsx",
}

OPENAI_CHAT_TIMEOUT  = 60
OPENAI_EMBED_TIMEOUT = 60
OPENAI_MAX_RETRIES   = 6
HF_EMBED_MODEL       = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ========= 2. Prompts / LLMs ========= #
PROMPT_QA = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Você é a SDR da AG Consultoria, chamada Gisa. Você é uma Analista de Finanças experiente e muito inteligente. "
        "Sua abordagem é sempre amigável e cordial, porém séria e confiável. Você precisa transmitir conhecimento e "
        "segurança ao explicar os indicadores financeiros dos clientes. Use os trechos abaixo para responder com exatidão.\n"
        "Se a resposta estiver em tabela, extraia o valor direto.\n"
        "Responda sempre informando mês e ano explícitos quando a pergunta pedir.\n\n"
        "Pergunta: {question}\n"
        "========\n{context}\n========\nResposta:"
    ),
)

PROMPT_VALID = PromptTemplate(
    input_variables=["question", "draft_answer", "sources", "hoje", "n_sources"],
    template=(
        "Você é um auditor de dados financeiros. Hoje é {hoje}.\n\n"
        "### Pergunta\n{question}\n\n"
        "### Resposta provisória\n{draft_answer}\n\n"
        "### Fontes (N={n_sources})\n{sources}\n\n"
        "Instruções de auditoria:\n"
        "- Só aprove a resposta se:\n"
        "  (a) houver pelo menos 1 fonte relevante (N>=1),\n"
        "  (b) os números mencionados puderem ser conferidos nos trechos,\n"
        "  (c) mês/ano citados na resposta forem compatíveis com os da pergunta quando especificados.\n"
        "- Se estiver tudo OK, devolva **apenas** PASS.\n"
        "- Se houver qualquer problema (sem fontes, mês/ano trocados, número não evidente nas fontes, etc.), "
        "refaça a resposta CORRETA usando exclusivamente as fontes listadas, em português claro, "
        "informando mês/ano e o valor exato.\n"
        "- Não explique o processo; devolva só a resposta final ou PASS."
    ),
)

PROMPT_VARIACAO = PromptTemplate(
    input_variables=["dados", "question"],
    template=(
        "Você é uma analista financeira. Abaixo estão trechos extraídos (podem conter mais de um registro por mês). "
        "Sua tarefa é CALCULAR a variação percentual entre os meses mencionados na pergunta.\n\n"
        "REGRAS:\n"
        "- Encontre para cada mês o valor de referência (Saldo/Valor) mais diretamente aplicável ao indicador pedido.\n"
        "- Se houver múltiplas linhas do mesmo mês, use a mais diretamente relacionada ao indicador (ex.: 'faturamento').\n"
        "- Fórmula: variação % = (valor_mes2 - valor_mes1) / valor_mes1 * 100.\n"
        "- Mencione os dois valores absolutos (com R$ quando for monetário) e a variação em % com 2 casas decimais.\n"
        "- Diga se foi alta ou queda. Não mostre passos de cálculo, apenas valores e conclusão.\n"
        "- Se faltar algum dos meses, informe objetivamente qual faltou.\n\n"
        "Pergunta: {question}\n"
        "=== DADOS ===\n{dados}\n=== FIM DOS DADOS ===\n\n"
        "Resposta final (curta e objetiva):"
    ),
)

# LLM primário (OpenAI) e fallback (Gemini)
LLM_QA_OPENAI    = ChatOpenAI(model="gpt-4o-mini", temperature=0.5,
                              timeout=OPENAI_CHAT_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)
LLM_VALID_OPENAI = ChatOpenAI(model="gpt-4o-mini", temperature=0.5,
                              timeout=OPENAI_CHAT_TIMEOUT, max_retries=OPENAI_MAX_RETRIES)

# Escolha o modelo Gemini conforme preferência/custo (flash é rápido/barato)
LLM_QA_VERTEX    = ChatVertexAI(model="gemini-1.5-flash", temperature=0.5)
LLM_VALID_VERTEX = ChatVertexAI(model="gemini-1.5-flash", temperature=0.5)

# Por padrão, usamos OpenAI e validamos com OpenAI; VALIDATOR fica com fallback manual (abaixo).
VALIDATOR_OPENAI = PROMPT_VALID | LLM_VALID_OPENAI
VALIDATOR_VERTEX = PROMPT_VALID | LLM_VALID_VERTEX

# ========= 3. Calendário / Meses ========= #
MONTH_ALIAS = {
    "Jan": "Janeiro", "Fev": "Fevereiro", "Mar": "Março", "Abr": "Abril",
    "Mai": "Maio", "Jun": "Junho", "Jul": "Julho", "Ago": "Agosto",
    "Set": "Setembro", "Out": "Outubro", "Nov": "Novembro", "Dez": "Dezembro",
}
MONTH_PT = {
    1: ("Janeiro", "Jan"),  2: ("Fevereiro", "Fev"), 3: ("Março", "Mar"),
    4: ("Abril", "Abr"),    5: ("Maio", "Mai"),      6: ("Junho", "Jun"),
    7: ("Julho", "Jul"),    8: ("Agosto", "Ago"),    9: ("Setembro", "Set"),
    10:("Outubro", "Out"), 11:("Novembro", "Nov"),  12:("Dezembro", "Dez"),
}
MONTH_ABBR_TO_NUM = {abbr: n for n, (_full, abbr) in MONTH_PT.items()}
MONTH_NUM_TO_ABBR = {n: abbr for n, (_full, abbr) in MONTH_PT.items()}
MONTH_NAME_TO_ABBR = {
    "janeiro": "Jan", "fevereiro": "Fev", "março": "Mar", "marco": "Mar",
    "abril": "Abr", "maio": "Mai", "junho": "Jun", "julho": "Jul",
    "agosto": "Ago", "setembro": "Set", "outubro": "Out",
    "novembro": "Nov", "dezembro": "Dez",
    "jan": "Jan","fev":"Fev","mar":"Mar","abr":"Abr","mai":"Mai","jun":"Jun",
    "jul":"Jul","ago":"Ago","set":"Set","out":"Out","nov":"Nov","dez":"Dez",
}


def run_llm_analytics(question: str, dados: str) -> str:
    # Tenta OpenAI primeiro; se der timeout, usa Gemini.
    try:
        return (PROMPT_VARIACAO | LLM_QA_OPENAI).invoke({"question": question, "dados": dados}).content
    except APITimeoutError:
        return (PROMPT_VARIACAO | LLM_QA_VERTEX).invoke({"question": question, "dados": dados}).content


def _get_docs(vector, question: str, flt, is_vertex: bool, k: int = 12) -> List[Document]:
    retriever = vector.as_retriever(search_kwargs={"k": k, "filter": flt})
    try:
        return retriever.invoke(question)  # API nova do LangChain
    except Exception:
        return retriever.get_relevant_documents(question)  # fallback


def retrieve_month_docs(vector, question: str, empresa_id: int, mes_abbr: str, ano: int, tipo: Optional[str]):
    vflt = build_vertex_filter(empresa_id, [mes_abbr], ano, tipo)
    docs = _get_docs(vector, question, vflt, is_vertex=_is_vertex_store(vector))
    if docs: return docs
    # sem tipo (fallback)
    vflt2 = build_vertex_filter(empresa_id, [mes_abbr], ano, None)
    return _get_docs(vector, question, vflt2, is_vertex=_is_vertex_store(vector))


def normalize_tipo(txt: Optional[str]) -> Optional[str]:
    if not txt: return None
    s = txt.lower().strip()
    s = s.replace("í", "i").replace("ó", "o").replace("é", "e").replace("á", "a").replace("â", "a").replace("ã", "a")
    s = s.replace(" ", "_").replace("-", "_")
    if s in {"ticket", "ticket_medio", "ticketmedio", "ticket_médio"}:
        return "ticket_medio"
    if s in {"faturamento", "faturacao"}:
        return "faturamento"
    if s in {"demanda"}:
        return "demanda"
    return s

def parse_number_br(text: str) -> Optional[float]:
    # Prioriza linhas com 'Saldo' ou 'Valor'
    prefer = [ln for ln in text.splitlines() if re.search(r"(saldo|valor)", ln.lower())]
    areas = prefer if prefer else [text]
    for area in areas:
        m = re.search(r"(?:(?:r\$\s*)?)((?:\d{1,3}\.)*\d+(?:,\d{1,2})?)", area.lower())
        if m:
            raw = m.group(1)
            return float(raw.replace(".", "").replace(",", "."))
    return None

def choose_record_value(docs: List[Document], tipo_alvo: Optional[str]) -> Optional[float]:
    alvo = normalize_tipo(tipo_alvo) if tipo_alvo else None
    # 1) metadado tipo == alvo
    for d in docs:
        t = normalize_tipo((d.metadata or {}).get("tipo"))
        if alvo and t == alvo:
            v = parse_number_br(d.page_content)
            if v is not None: return v
    # 2) conteúdo menciona o tipo
    if alvo:
        base = alvo.split("_")[0]
        for d in docs:
            if base in d.page_content.lower():
                v = parse_number_br(d.page_content)
                if v is not None: return v
    # 3) primeiro número parsável
    for d in docs:
        v = parse_number_br(d.page_content)
        if v is not None: return v
    return None



def answer_variacao(vector, file_path: str, pergunta: str, empresa_id: int, meses_abbr: List[str], ano: int, tipo: Optional[str]):
    mes1, mes2 = meses_abbr[0], meses_abbr[1]
    docs1 = retrieve_month_docs(vector, pergunta, empresa_id, mes1, ano, tipo)
    docs2 = retrieve_month_docs(vector, pergunta, empresa_id, mes2, ano, tipo)

    if not docs1 or not docs2:
        faltou = []
        if not docs1: faltou.append(f"{mes1}/{ano}")
        if not docs2: faltou.append(f"{mes2}/{ano}")
        return {
            "answer": f"Não foi possível calcular a variação: faltam dados para {', '.join(faltou)}.",
            "source_documents": (docs1 or []) + (docs2 or []),
        }, None

    v1 = choose_record_value(docs1, tipo)
    v2 = choose_record_value(docs2, tipo)
    if v1 is None or v2 is None or v1 == 0:
        motivo = "valor não encontrado" if (v1 is None or v2 is None) else "valor do mês base é zero"
        return {"answer": f"Não foi possível calcular a variação ({motivo}).", "source_documents": docs1 + docs2}, None

    var = (v2 - v1) / v1 * 100.0
    sinal = "alta" if var > 0 else "queda" if var < 0 else "estabilidade"

    def br_money(x): return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    resp = (
        f"Faturamento em {mes1}/{ano}: {br_money(v1)}\n"
        f"Faturamento em {mes2}/{ano}: {br_money(v2)}\n"
        f"Variação: {var:.2f}% ({sinal})."
    )
    return {"answer": resp, "source_documents": docs1 + docs2}, None



# ========= 4. Embeddings & Vetores ========= #
def get_openai_embeddings():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        timeout=OPENAI_EMBED_TIMEOUT,
        max_retries=OPENAI_MAX_RETRIES,
    )

def get_hf_embeddings():
    """
    Lazy import para não quebrar se pacotes locais não estiverem instalados.
    """
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception as e:
        raise RuntimeError(
            "Fallback local indisponível: instale 'sentence-transformers' e 'langchain-huggingface'."
        ) from e
    return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

# --- Vertex AI: criar/obter Index e Endpoint, e implantar se necessário --- #
def ensure_vertex_index_and_endpoint(
    project_id: str,
    region: str,
    bucket: str,
    index_display_name: str,
    endpoint_display_name: str,
    dimensions: int = 768,
    distance: str = "DOT_PRODUCT_DISTANCE",
) -> Tuple[str, str]:
    aiplatform.init(project=project_id, location=region, staging_bucket=bucket)

    # 1) Index
    idx_list = aiplatform.MatchingEngineIndex.list(filter=f'display_name="{index_display_name}"')
    if idx_list:
        index = idx_list[0]
    else:
        index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
            display_name=index_display_name,
            dimensions=dimensions,
            approximate_neighbors_count=150,
            distance_measure_type=distance,
            index_update_method="STREAM_UPDATE",
        )

    # 2) Endpoint
    ep_list = aiplatform.MatchingEngineIndexEndpoint.list(filter=f'display_name="{endpoint_display_name}"')
    if ep_list:
        endpoint = ep_list[0]
    else:
        endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
            display_name=endpoint_display_name,
            public_endpoint_enabled=True,
        )

    # 3) Deploy se ainda não estiver implantado
    endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint.name)
    deployed_ids = [d.index.split("/")[-1] for d in endpoint.gca_resource.deployed_indexes]
    idx_id = index.resource_name.split("/")[-1]
    if idx_id not in deployed_ids:
        dep_id = f"dep_{idx_id[:8]}"  # válido: começa com letra, usa [a-zA-Z0-9_]
        op = endpoint.deploy_index(index=index, deployed_index_id=dep_id)
        print("   Aguardando deploy concluir (pode levar ~20–40 min na 1ª vez)...")
        op.result(timeout=3600)
        print("   Deploy concluído.")

    index_id = index.resource_name.split("/")[-1]
    endpoint_id = endpoint.resource_name.split("/")[-1]
    return index_id, endpoint_id

# --- Construção/abertura de vetor Vertex + ingestão inicial --- #
def ensure_vector_vertexai(bucket: str, index_id: str, endpoint_id: str, file_path: str, empresa_id: int):
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=bucket)
    embedding = VertexAIEmbeddings(model_name="text-embedding-005")  # 768 dims

    store = VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_bucket_name=bucket.replace("gs://", ""),
        index_id=index_id,
        endpoint_id=endpoint_id,
        embedding=embedding,
        stream_update=True,
    )

    # Carrega e injeta documentos (metadados como strings p/ filtros textuais)
    loader_kind = choose_loader(file_path)
    docs = line_per_value_loader(file_path) if loader_kind == "line" else excel_table_loader(file_path, empresa_id)
    if docs:
        store.add_texts([d.page_content for d in docs], metadatas=[d.metadata for d in docs])
    return store

# --- (Fallback local) Construção/abertura de vetores Chroma/HF --- #
def ensure_vector_hf(file_path: str, empresa_id: int):
    stem = Path(file_path).stem
    store_dir = f"chromax_{stem}_hf"
    if os.path.exists(store_dir):
        return Chroma(persist_directory=store_dir, embedding_function=get_hf_embeddings())
    loader_kind = choose_loader(file_path)
    docs = line_per_value_loader(file_path) if loader_kind == "line" else excel_table_loader(file_path, empresa_id)
    return Chroma.from_documents(docs, embedding=get_hf_embeddings(), persist_directory=store_dir)

# ========= 5. Loaders ========= #
def _infer_tipo_from_context(context: str, sheet_name: str | None = None) -> Optional[str]:
    txt = f"{context or ''} || {sheet_name or ''}".lower()
    if "faturamento" in txt:
        return "faturamento"
    if "demanda" in txt:
        return "demanda"
    if "ticket" in txt or "tícket" in txt:
        return "ticket médio"
    return None

def excel_table_loader(path: str, empresa_id: int) -> List[Document]:
    dfs = pd.read_excel(path, sheet_name=None, header=None)
    docs: List[Document] = []
    for sheet_name, raw in dfs.items():
        first_row_idx = raw.first_valid_index()
        if first_row_idx is None:
            continue
        context = str(raw.iloc[first_row_idx, 0]).strip()

        header_idx = first_row_idx + 1
        header = [str(h).strip() for h in raw.iloc[header_idx]]

        data = raw.iloc[header_idx + 1 :].copy()
        data.columns = header
        data = data.dropna(how="all")

        tipo_ctx = _infer_tipo_from_context(context, sheet_name)

        if "Mês" not in data.columns:
            meta = {"sheet": sheet_name, "empresa_id": str(empresa_id)}
            if tipo_ctx:
                meta["tipo"] = tipo_ctx
            docs.append(Document(f"{context}\n{data.to_csv(index=False, sep='|')}", metadata=meta))
            continue

        anos_cols = [c for c in data.columns if re.fullmatch(r"\d{4}", str(c))]
        long_df = pd.melt(data, id_vars=["Mês"], value_vars=anos_cols,
                          var_name="Ano", value_name="Saldo").dropna(subset=["Saldo"])

        for _, row in long_df.iterrows():
            m_abrev = str(row["Mês"]).strip()
            m_full  = MONTH_ALIAS.get(m_abrev, m_abrev)
            ano, saldo = int(row["Ano"]), row["Saldo"]
            texto = f"Contexto: {context}\nMês: {m_full} ({m_abrev})\nAno: {ano}\nSaldo: {saldo}"
            meta = {"sheet": sheet_name, "mes": m_abrev, "ano": str(ano), "empresa_id": str(empresa_id)}
            if tipo_ctx:
                meta["tipo"] = tipo_ctx
            docs.append(Document(texto, metadata=meta))
    return docs

def line_per_value_loader(path: str) -> List[Document]:
    df = pd.read_excel(path)
    df.columns = [c.lower() for c in df.columns]
    req = {"mes_ano", "valor", "tipo", "empresa_id"}
    if not req.issubset(set(df.columns)):
        raise ValueError("Colunas esperadas não encontradas (mes_ano, valor, tipo, empresa_id).")
    docs: List[Document] = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row["mes_ano"])
        ano, mesN = int(dt.year), int(dt.month)
        mes_full, mes_abrev = MONTH_PT[mesN]
        valor = row["valor"]
        tipo = str(row["tipo"]).strip().lower()
        emp = str(int(row["empresa_id"]))  # string para filtros textuais
        texto = (
            f"Empresa ID: {emp}\n"
            f"Tipo de indicador: {tipo}\n"
            f"Ano: {ano}\n"
            f"Mês: {mes_full} ({mes_abrev})\n"
            f"Valor: {valor}"
        )
        docs.append(Document(texto, metadata={
            "empresa_id": emp, "tipo": tipo, "ano": str(ano), "mes": mes_abrev
        }))
    return docs

def choose_loader(path: str) -> str:
    df = pd.read_excel(path, nrows=5)
    cols = {c.lower() for c in df.columns}
    return "line" if {"mes_ano", "valor", "tipo", "empresa_id"}.issubset(cols) else "cross"

# ========= 6. Parse indicador / mês / ano ========= #
def parse_indicador(pergunta: str) -> Optional[str]:
    s = pergunta.lower()
    if "faturamento" in s:
        return "faturamento"
    if "demanda" in s:
        return "demanda"
    if "ticket médio" in s or "ticket medio" in s or "tícket médio" in s or "tícket medio" in s or "ticket" in s:
        return "ticket médio"
    return None

def parse_meses_ano(pergunta: str):
    """
    Retorna (lista_meses_abrev, ano_int) quando encontrado; caso contrário ([], None).
    Agora entende: "maio e junho de 2025", "de maio para junho de 2025",
    "entre maio e junho de 2025", "maio a junho de 2025", "maio vs junho de 2025".
    """
    s = pergunta.lower()

    # 1) "maio e junho de 2025"
    m_pair = re.search(
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*e\s*"
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*(?:de)?\s*(\d{4})",
        s
    )
    if m_pair:
        mes1_txt, mes2_txt, ano = m_pair.group(1), m_pair.group(2), int(m_pair.group(3))
        m1 = MONTH_NAME_TO_ABBR.get(mes1_txt); m2 = MONTH_NAME_TO_ABBR.get(mes2_txt)
        meses = [m for m in (m1, m2) if m]
        return (meses, ano) if meses else ([], None)

    # 2) "de maio para junho de 2025" | "entre maio e junho de 2025" | "maio a junho de 2025" | "maio vs junho de 2025"
    m_pair_var = re.search(
        r"(?:de|entre)?\s*"
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*"
        r"(?:para|e|a|vs|x)\s*"
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*(?:de)?\s*(\d{4})",
        s
    )
    if m_pair_var:
        mes1_txt, mes2_txt, ano = m_pair_var.group(1), m_pair_var.group(2), int(m_pair_var.group(3))
        m1 = MONTH_NAME_TO_ABBR.get(mes1_txt); m2 = MONTH_NAME_TO_ABBR.get(mes2_txt)
        meses = [m for m in (m1, m2) if m]
        return (meses, ano) if meses else ([], None)

    # (demais padrões que você já tinha, mantidos)
    m_pair_dup = re.search(
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*(?:de)?\s*(\d{4})\s*e\s*"
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*(?:de)?\s*(\d{4})",
        s
    )
    if m_pair_dup:
        mes1_txt, ano1, mes2_txt, ano2 = m_pair_dup.group(1), int(m_pair_dup.group(2)), m_pair_dup.group(3), int(m_pair_dup.group(4))
        if ano1 == ano2:
            m1 = MONTH_NAME_TO_ABBR.get(mes1_txt); m2 = MONTH_NAME_TO_ABBR.get(mes2_txt)
            meses = [m for m in (m1, m2) if m]
            return (meses, ano1) if meses else ([], None)

    m_pair_abbr = re.findall(r"(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[-/](\d{4})", s)
    if len(m_pair_abbr) >= 2:
        (m1_txt, a1), (m2_txt, a2) = m_pair_abbr[0], m_pair_abbr[1]
        if a1 == a2:
            m1 = MONTH_NAME_TO_ABBR.get(m1_txt); m2 = MONTH_NAME_TO_ABBR.get(m2_txt)
            meses = [m for m in (m1, m2) if m]
            return (meses, int(a1)) if meses else ([], None)

    m_pair_num = re.findall(r"\b(1[0-2]|0?[1-9])[/-](\d{4})\b", s)
    if len(m_pair_num) >= 2:
        (m1n, a1), (m2n, a2) = m_pair_num[0], m_pair_num[1]
        if a1 == a2:
            _, m1 = MONTH_PT[int(m1n)]; _, m2 = MONTH_PT[int(m2n)]
            return ([m1, m2], int(a1))

    # único mês/ano (mantido)
    m1 = re.search(r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*(?:de)?\s*(\d{4})", s)
    if m1:
        mes_txt, ano = m1.group(1), int(m1.group(2))
        m = MONTH_NAME_TO_ABBR.get(mes_txt);  return ([m], ano) if m else ([], None)

    m2 = re.search(r"(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[-/](\d{4})", s)
    if m2:
        m = MONTH_NAME_TO_ABBR.get(m2.group(1)); return ([m], int(m2.group(2))) if m else ([], None)

    m3 = re.search(r"\b(1[0-2]|0?[1-9])[/-](\d{4})\b", s)
    if m3:
        _, m = MONTH_PT[int(m3.group(1))]; return ([m], int(m3.group(2)))

    return ([], None)



# ========= 7. Filtros (Chroma vs Vertex) ========= #
def build_chroma_filter(empresa_id: int, meses_abbr: List[str], ano: Optional[int], tipo: Optional[str]):
    # Metadatas foram salvas como strings nos loaders; use strings no filtro também.
    clauses = [{"empresa_id": {"$eq": str(empresa_id)}}]
    if ano is not None:
        clauses.append({"ano": {"$eq": str(ano)}})
    if tipo:
        clauses.append({"tipo": {"$eq": tipo}})
    if meses_abbr:
        if len(meses_abbr) == 1:
            clauses.append({"mes": {"$eq": meses_abbr[0]}})
        else:
            clauses.append({"$or": [{"mes": {"$eq": m}} for m in meses_abbr]})
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}

def build_vertex_filter(empresa_id: int, meses_abbr: List[str], ano: Optional[int], tipo: Optional[str]) -> List[Namespace]:
    """
    Vertex AI Vector Search espera uma lista de Namespace (filtro textual).
    AND entre Namespaces; OR dentro de allow_tokens.
    """
    filters: List[Namespace] = [Namespace(name="empresa_id", allow_tokens=[str(empresa_id)])]
    if ano is not None:
        filters.append(Namespace(name="ano", allow_tokens=[str(ano)]))
    if tipo:
        filters.append(Namespace(name="tipo", allow_tokens=[tipo]))
    if meses_abbr:
        filters.append(Namespace(name="mes", allow_tokens=meses_abbr))  # OR entre meses
    return filters

# ========= 8. Cadeias e fallback ========= #
def make_chain_no_memory(retriever, llm):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,
        combine_docs_chain_kwargs={"prompt": PROMPT_QA},
        return_source_documents=True,
    )

def make_chain(retriever, memory, llm):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT_QA},
        return_source_documents=True,
    )

def validar_resposta(pergunta: str, provisoria: str, fontes_docs: List[Document]) -> str:
    fontes_txt = "\n".join(doc.page_content for doc in fontes_docs)
    hoje = datetime.now().strftime("%d/%m/%Y")
    n_sources = len(fontes_docs)
    # tenta OpenAI; se der timeout, valida com Gemini
    try:
        feedback = (PROMPT_VALID | LLM_VALID_OPENAI).invoke(
            {"question": pergunta, "draft_answer": provisoria, "sources": fontes_txt, "hoje": hoje, "n_sources": str(n_sources)}
        ).content.strip()
    except APITimeoutError:
        feedback = (PROMPT_VALID | LLM_VALID_VERTEX).invoke(
            {"question": pergunta, "draft_answer": provisoria, "sources": fontes_txt, "hoje": hoje, "n_sources": str(n_sources)}
        ).content.strip()
    return provisoria if feedback == "PASS" else feedback

def current_month_year():
    now = datetime.now()
    return MONTH_PT[now.month][1], now.year

def previous_month(m_abbr: str, ano: int):
    m_num = MONTH_ABBR_TO_NUM[m_abbr]
    m_num -= 1
    if m_num == 0:
        return "Dez", ano - 1
    return MONTH_NUM_TO_ABBR[m_num], ano

def _is_vertex_store(store) -> bool:
    return isinstance(store, VectorSearchVectorStore)

def invoke_with_fallback(vector_primary, file_path, empresa_id, memory, question,
                         chroma_filter, vertex_filter):
    """
    Ordem de fallback:
    1) Retriever atual + LLM OpenAI
    2) Mesmo retriever + LLM Gemini (Vertex)
    3) Fallback local (Chroma+HF) + LLM Gemini (se libs instaladas)

    IMPORTANTE: quando memory é None, usamos a cadeia "no-memory"
    e passamos 'chat_history': [] no invoke().
    """
    # Escolhe retriever conforme tipo de store
    if _is_vertex_store(vector_primary):
        retriever = vector_primary.as_retriever(search_kwargs={"k": 8, "filter": vertex_filter})
    else:
        retriever = vector_primary.as_retriever(search_kwargs={"k": 8, "filter": chroma_filter})

    def _build_chain(retr, mem, llm):
        return make_chain(retr, mem, llm) if mem is not None else make_chain_no_memory(retr, llm)

    def _invoke_chain(chain, mem):
        # Se não há memória, o ConversationalRetrievalChain exige 'chat_history' explicitamente.
        if mem is None:
            return chain.invoke({"question": question, "chat_history": []})
        else:
            return chain.invoke({"question": question})

    # 1) tenta com OpenAI
    qa = _build_chain(retriever, memory, LLM_QA_OPENAI)
    try:
        return _invoke_chain(qa, memory)
    except APITimeoutError:
        pass

    # 2) tenta com Gemini (sem trocar de vetor)
    qa_gemini = _build_chain(retriever, memory, LLM_QA_VERTEX)
    try:
        return _invoke_chain(qa_gemini, memory)
    except Exception:
        # Pode falhar por outros motivos; seguimos pro fallback local
        pass

    # 3) fallback local (Chroma + HF) + Gemini
    try:
        vector_fallback = ensure_vector_hf(file_path, empresa_id)
        retriever2 = vector_fallback.as_retriever(search_kwargs={"k": 8, "filter": chroma_filter})
        qa2 = make_chain_no_memory(retriever2, llm=LLM_QA_VERTEX)
        res2 = qa2.invoke({"question": question, "chat_history": []})
        if "answer" in res2 and isinstance(res2["answer"], str):
            res2["answer"] += "\n\n(Nota: usei índice local por indisponibilidade temporária da API/LLM.)"
        return res2
    except Exception as e:
        # Último recurso: retornar erro legível
        return {"answer": f"Não consegui concluir por falha de rede/LLM e fallback local indisponível: {e}", "source_documents": []}

def list_available_indicators(vector, empresa_id: int, candidates=None) -> List[str]:
    if candidates is None:
        candidates = ["faturamento", "demanda", "ticket_medio"]
    found = []
    for t in candidates:
        vflt = build_vertex_filter(empresa_id, [], None, t)
        docs = _get_docs(vector, f"listar {t}", vflt, is_vertex=_is_vertex_store(vector), k=1)
        if docs:
            found.append(t)
    return found

def answer_with_fallback(file_path: str, vector, memory, pergunta: str, empresa_id: int,
                         tipo: Optional[str], meses_abbr: List[str], ano: Optional[int]):
    assumiu_mes_atual = False
    if not meses_abbr:
        m_now, y_now = current_month_year()
        meses_abbr = [m_now]
        ano = y_now if ano is None else ano
        assumiu_mes_atual = True

    chroma_flt = build_chroma_filter(empresa_id, meses_abbr, ano, tipo)
    vertex_flt = build_vertex_filter(empresa_id, meses_abbr, ano, tipo)

    intent_var = any(x in pergunta.lower() for x in ["varia", "percent", "diferen", "compar", "vs", "para", "a"])
    if intent_var and len(meses_abbr) == 2 and ano is not None:
        return answer_variacao(vector, file_path, pergunta, empresa_id, meses_abbr, ano, tipo)

    # A: com tipo (se houver)
    res = invoke_with_fallback(vector, file_path, empresa_id, memory, pergunta, chroma_flt, vertex_flt)
    fontes = res.get("source_documents", []) or []
    if fontes:
        return res, None

    # B: sem tipo
    if tipo:
        chroma_no_tipo = build_chroma_filter(empresa_id, meses_abbr, ano, None)
        vertex_no_tipo = build_vertex_filter(empresa_id, meses_abbr, ano, None)
        res2 = invoke_with_fallback(vector, file_path, empresa_id, None, pergunta, chroma_no_tipo, vertex_no_tipo)
        fontes2 = res2.get("source_documents", []) or []
        if fontes2:
            info = "(Obs.: o indicador não estava marcado no índice; relaxei o filtro de 'tipo'.)"
            return res2, info

    # Recuo mês a mês (se assumiu mês atual)
    if not assumiu_mes_atual or len(meses_abbr) > 1:
        return res, None

    m_try, y_try = meses_abbr[0], ano
    for _ in range(24):
        m_try, y_try = previous_month(m_try, y_try)

        if tipo:
            chroma_fb_tipo = build_chroma_filter(empresa_id, [m_try], y_try, tipo)
            vertex_fb_tipo = build_vertex_filter(empresa_id, [m_try], y_try, tipo)
            res_fb_tipo = invoke_with_fallback(vector, file_path, empresa_id, None, pergunta, chroma_fb_tipo, vertex_fb_tipo)
            fontes_fb_tipo = res_fb_tipo.get("source_documents", []) or []
            if fontes_fb_tipo:
                info = f"(Sem dados para {meses_abbr[0]}/{ano}. Mostrando o último disponível: {m_try}/{y_try}.)"
                return res_fb_tipo, info

        chroma_fb = build_chroma_filter(empresa_id, [m_try], y_try, None)
        vertex_fb = build_vertex_filter(empresa_id, [m_try], y_try, None)
        res_fb = invoke_with_fallback(vector, file_path, empresa_id, None, pergunta, chroma_fb, vertex_fb)
        fontes_fb = res_fb.get("source_documents", []) or []
        if fontes_fb:
            info = f"(Sem dados para {meses_abbr[0]}/{ano}. Mostrando o último disponível: {m_try}/{y_try}.)"
            return res_fb, info

    return {"answer": "", "source_documents": []}, "Não encontrei dados para o mês atual nem para meses anteriores (até 24 meses)."

# ========= 9. Vetor + memória (construtor principal) ========= #
def build_vector_and_memory(file_path: str, empresa_id: int, bucket: str, index_id: str, endpoint_id: str):
    vector_primary = ensure_vector_vertexai(
        bucket=bucket,
        index_id=index_id,
        endpoint_id=endpoint_id,
        file_path=file_path,
        empresa_id=empresa_id
    )

    memory = ConversationSummaryBufferMemory(
        llm=LLM_QA_OPENAI,  # o summarizer tentará OpenAI; se quiser dá p/ trocar por Gemini
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        max_token_limit=1500,
    )
    return vector_primary, memory

# ========= 10. CLI ========= #
if __name__ == "__main__":
    file_path = BASES["5"]  # troque para "4" se quiser
    # shutil.rmtree(f"chromax_{Path(file_path).stem}_hf", ignore_errors=True)

    print("✔ Inicializando Vertex AI e garantindo Index/Endpoint...")
    index_id, endpoint_id = ensure_vertex_index_and_endpoint(
        project_id=PROJECT_ID,
        region=REGION,
        bucket=BUCKET,
        index_display_name=INDEX_DISPLAY_NAME,
        endpoint_display_name=ENDPOINT_DISPLAY_NAME,
        dimensions=VECTOR_DIMENSIONS,
        distance="DOT_PRODUCT_DISTANCE",
    )
    print(f"   Index ID: {index_id}")
    print(f"   Endpoint ID: {endpoint_id}\n")

    empresa_id = int(input("empresa_id para esta sessão: ").strip())
    vector, memory = build_vector_and_memory(file_path, empresa_id, BUCKET, index_id, endpoint_id)
    print("✅ Base carregada no Vertex AI Vector Search!\n")

    while True:
        pergunta = input("Pergunta (ou 'sair'): ").strip()
        if pergunta.lower() == "sair":
            break

        tipo = parse_indicador(pergunta)
        meses_abbr, ano = parse_meses_ano(pergunta)

        res, fallback_info = answer_with_fallback(file_path, vector, memory, pergunta, empresa_id, tipo, meses_abbr, ano)
        provisoria = res.get("answer", "")
        fontes = res.get("source_documents", []) or []

        if len(fontes) == 0:
            msg = (
                "Não encontrei dados compatíveis com os filtros "
                f"(empresa_id={empresa_id}"
                f"{', meses='+','.join(meses_abbr) if meses_abbr else ''}"
                f"{', ano='+str(ano) if ano else ''}"
                f"{', tipo='+tipo if tipo else ''}). "
                "Verifique se o arquivo contém esse período/empresa/indicador."
            )
            print("\n🤖 " + msg + "\n")
            continue

        final = validar_resposta(
            pergunta if not fallback_info else f"{pergunta}\n{fallback_info}",
            provisoria,
            fontes
        )
        print("\n🤖", final, "\n")
