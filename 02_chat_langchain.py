# 02_chat_langchain_fix.py
# Aplicação (Servir o Modelo): recebe perguntas, busca no Vertex Vector Search e responde com LLM.
# Não realiza ingestão. Usa índice/endpoint já existentes (criados pelo 01_ingest.py).

from __future__ import annotations

import os, re, json, warnings, time
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import pandas as pd
from dotenv import load_dotenv

# LangChain / Vertex
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings, VectorSearchVectorStore

# GCP Vertex AI
from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace

# (Opcional) Fallback local
USE_LOCAL_FALLBACK = False
try:
    from langchain_chroma import Chroma
    USE_LOCAL_FALLBACK = True
except Exception:
    pass

# (Opcional) OpenAI fallback para LLM
USE_OPENAI_FALLBACK = bool(os.getenv("USE_OPENAI_FALLBACK", "").strip())
if USE_OPENAI_FALLBACK:
    try:
        from langchain_openai import ChatOpenAI
    except Exception:
        USE_OPENAI_FALLBACK = False

# ======== ENV ======== #
load_dotenv()

PROJECT_ID   = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
REGION       = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1").strip()
BUCKET       = os.getenv("VERTEX_STAGING_BUCKET", "").strip()

INDEX_NAME   = os.getenv("VERTEX_INDEX_NAME", "indices-financeiros-gisa-v1").strip()
ENDPOINT_NAME= os.getenv("VERTEX_ENDPOINT_NAME", "endpoint-financeiros-gisa-v1").strip()

GEMINI_QA_MODEL    = os.getenv("GEMINI_QA_MODEL", "gemini-2.5-flash").strip()
GEMINI_VALID_MODEL = os.getenv("GEMINI_VALID_MODEL", "gemini-2.5-flash").strip()

# Permite ignorar .ingest_state.json para sempre pegar o índice mais novo por prefixo
IGNORE_STATE = os.getenv("VERTEX_IGNORE_STATE", "").strip() == "1"

# Evita warning do model_garden
warnings.filterwarnings("ignore", category=UserWarning, module="vertexai._model_garden._model_garden_models")

# ======== PROMPTS ======== #
PROMPT_QA = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Você é a SDR da AG Consultoria, chamada Gisa. Você é uma Analista de Finanças experiente e muito inteligente. "
        "Sua abordagem é sempre amigável e cordial, porém séria e confiável. "
        "Use os trechos abaixo para responder com exatidão.\n"
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
        "- Se houver qualquer problema, refaça a resposta CORRETA usando exclusivamente as fontes listadas, em português claro, "
        "informando mês/ano e o valor exato.\n"
        "- Não explique o processo; devolva só a resposta final ou PASS."
    ),
)

# ======== Meses / Datas ======== #
MONTH_PT = {
    1: ("Janeiro", "Jan"),  2: ("Fevereiro", "Fev"), 3: ("Março", "Mar"),
    4: ("Abril", "Abr"),    5: ("Maio", "Mai"),      6: ("Junho", "Jun"),
    7: ("Julho", "Jul"),    8: ("Agosto", "Ago"),    9: ("Setembro", "Set"),
    10:("Outubro", "Out"), 11:("Novembro", "Nov"),  12:("Dezembro", "Dez"),
}
MONTH_NAME_TO_ABBR = {
    "janeiro": "Jan", "fevereiro": "Fev", "março": "Mar", "marco": "Mar",
    "abril": "Abr", "maio": "Mai", "junho": "Jun", "julho": "Jul",
    "agosto": "Ago", "setembro": "Set", "outubro": "Out",
    "novembro": "Nov", "dezembro": "Dez",
    "jan": "Jan","fev":"Fev","mar":"Mar","abr":"Abr","mai":"Mai","jun":"Jun",
    "jul":"Jul","ago":"Ago","set":"Set","out":"Out","nov":"Nov","dez":"Dez",
}
MONTH_ABBR_TO_NUM = {abbr: n for n, (_full, abbr) in MONTH_PT.items()}
MONTH_NUM_TO_ABBR = {n: abbr for n, (_full, abbr) in MONTH_PT.items()}

# ======== Utilidades ======== #
def current_month_year():
    now = datetime.now()
    return MONTH_PT[now.month][1], now.year

def previous_month(m_abbr: str, ano: int):
    m_num = MONTH_ABBR_TO_NUM[m_abbr]
    m_num -= 1
    if m_num == 0:
        return "Dez", ano - 1
    return MONTH_NUM_TO_ABBR[m_num], ano

def normalize_tipo(txt: Optional[str]) -> Optional[str]:
    if not txt: return None
    s = str(txt).lower().strip()
    s = (s.replace("í","i").replace("ó","o").replace("é","e")
           .replace("á","a").replace("â","a").replace("ã","a"))
    s = s.replace(" ", "_").replace("-", "_")
    if s in {"ticket","ticket_medio","ticketmedio","ticket_médio"}: return "ticket_medio"
    if s in {"faturamento","faturacao"}: return "faturamento"
    if s in {"demanda"}: return "demanda"
    return s

def parse_indicador(pergunta: str) -> Optional[str]:
    s = pergunta.lower()
    if "faturamento" in s:
        return "faturamento"
    if "demanda" in s:
        return "demanda"
    if "ticket" in s:  # cobre "ticket medio", "ticket médio" etc
        return "ticket_medio"  # já retorna normalizado
    return None

def parse_meses_ano(pergunta: str):
    s = pergunta.lower().strip()

    month_pat = r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)"
    conn_pat  = r"(?:entre\s+|de\s+|a\s+|até\s+|ate\s+|para\s+|vs\s+|x\s+|e\s+|para\s+)"

    # --- 1) Pares com nome do mês e ano opcional em cada lado ---
    m = re.search(
        rf"(?P<m1>{month_pat})(?:\s*de\s*(?P<y1>\d{{4}}))?\s*{conn_pat}\s*(?P<m2>{month_pat})(?:\s*de\s*(?P<y2>\d{{4}}))?",
        s
    )
    if m:
        m1_txt, m2_txt = m.group("m1"), m.group("m2")
        y1, y2 = m.group("y1"), m.group("y2")
        m1 = MONTH_NAME_TO_ABBR.get(m1_txt)
        m2 = MONTH_NAME_TO_ABBR.get(m2_txt)
        if m1 and m2:
            # mesmo ano explícito nos dois lados
            if y1 and y2 and y1 == y2:
                return ([m1, m2], int(y1))
            # ano só de um lado => assume para ambos
            if (y1 and not y2):
                return ([m1, m2], int(y1))
            if (y2 and not y1):
                return ([m1, m2], int(y2))
            # nenhum ano → deixa o caller decidir (retorna sem ano)
            return ([m1, m2], None)

    # --- 2) Pares no formato "MM/AAAA ... MM/AAAA" ou "mmm/AAAA ... mmm/AAAA" ---
    m = re.search(
        r"(?P<m1>(?:\d{1,2}|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez))\s*/\s*(?P<y1>\d{4})\s*"
        rf"{conn_pat}"
        r"\s*(?P<m2>(?:\d{1,2}|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez))\s*/\s*(?P<y2>\d{4})",
        s
    )
    if m:
        def to_abbr(mtxt: str) -> Optional[str]:
            mtxt = mtxt.strip().lower()
            if mtxt.isdigit():
                n = int(mtxt)
                if 1 <= n <= 12:
                    return MONTH_NUM_TO_ABBR[n]
                return None
            mtxt = mtxt[:3]
            cap = mtxt.capitalize()
            return cap if cap in MONTH_ABBR_TO_NUM else None

        m1_raw, y1 = m.group("m1"), int(m.group("y1"))
        m2_raw, y2 = m.group("m2"), int(m.group("y2"))
        m1 = to_abbr(m1_raw)
        m2 = to_abbr(m2_raw)
        if m1 and m2 and y1 == y2:
            return ([m1, m2], y1)

    # --- 3) Um mês explícito "mês de AAAA" ---
    m1 = re.search(rf"{month_pat}\s*(?:de)?\s*(\d{{4}})", s)
    if m1:
        mes_txt, ano = m1.group(0), int(re.search(r"\d{4}", m1.group(0)).group(0))
        # extrai o nome do mês propriamente
        mes_nome = re.search(month_pat, mes_txt).group(0)
        m = MONTH_NAME_TO_ABBR.get(mes_nome)
        if m: 
            return ([m], ano)

    # --- 4) Um mês no formato "MM/AAAA" ou "mmm/AAAA" ---
    m1 = re.search(r"(?:\b(\d{1,2}|jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)\s*/\s*(\d{4}))", s)
    if m1:
        m_raw, ano = m1.group(1).lower(), int(m1.group(2))
        if m_raw.isdigit():
            n = int(m_raw)
            if 1 <= n <= 12:
                return ([MONTH_NUM_TO_ABBR[n]], ano)
        else:
            ab = m_raw[:3].capitalize()
            if ab in MONTH_ABBR_TO_NUM:
                return ([ab], ano)

    return ([], None)

def build_vertex_filter(empresa_id: int, meses_abbr: List[str], ano: Optional[int], tipo: Optional[str]) -> List[Namespace]:
    filters: List[Namespace] = [Namespace(name="empresa_id", allow_tokens=[str(empresa_id)])]
    if ano is not None:
        filters.append(Namespace(name="ano", allow_tokens=[str(ano)]))
    if tipo:
        tipo = normalize_tipo(tipo)  # normaliza SEMPRE
        filters.append(Namespace(name="tipo", allow_tokens=[tipo]))
    if meses_abbr:
        filters.append(Namespace(name="mes", allow_tokens=meses_abbr))
    return filters

def parse_number_br(text: str) -> Optional[float]:
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
    for d in docs:
        t = normalize_tipo((d.metadata or {}).get("tipo"))
        if alvo and t == alvo:
            v = parse_number_br(d.page_content)
            if v is not None: return v
    if alvo:
        base = alvo.split("_")[0]
        for d in docs:
            if base in d.page_content.lower():
                v = parse_number_br(d.page_content)
                if v is not None: return v
    for d in docs:
        v = parse_number_br(d.page_content)
        if v is not None: return v
    return None

# ======== Intenção de série histórica / sazonalidade ======== #
def is_time_series_intent(q: str) -> bool:
    ql = q.lower()
    gatilhos = [
        "sazonalidade", "padrões", "padroes", "tendência", "tendencias",
        "ciclo", "histórico", "historico", "evolução", "evolucao",
        "ao longo", "últimos 12", "ultimos 12", "mês a mês", "mes a mes",
        "variações mensais", "serie historica", "série historica",
    ]
    return any(g in ql for g in gatilhos)

def fetch_timeseries_docs(store, empresa_id: int, tipo: Optional[str], k: int = 200):
    # Filtro amplo: empresa (+ tipo se houver) — sem 'mes' nem 'ano'
    vflt = build_vertex_filter(empresa_id, meses_abbr=[], ano=None, tipo=tipo)
    retriever = store.as_retriever(search_kwargs={"k": k, "filter": vflt})
    q = "histórico mensal do indicador com mês, ano e valor"
    try:
        return retriever.invoke(q) or []
    except Exception:
        try:
            return retriever.get_relevant_documents(q) or []
        except Exception:
            return []

def docs_to_timeseries(docs: List[Document]) -> pd.DataFrame:
    rows = []
    for d in docs:
        md = d.metadata or {}
        mes = md.get("mes")
        ano = md.get("ano")
        val = parse_number_br(d.page_content)
        if mes and ano and val is not None:
            mnum = MONTH_ABBR_TO_NUM.get(mes)
            if mnum:
                rows.append({"ano": int(ano), "mes": mnum, "valor": float(val)})
    if not rows:
        return pd.DataFrame(columns=["ano","mes","valor","mes_ano"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["ano","mes"])
    df["mes_ano"] = pd.to_datetime(dict(year=df["ano"], month=df["mes"], day=1))
    return df.sort_values("mes_ano").reset_index(drop=True)

def analyze_seasonality(df: pd.DataFrame, tipo_label: Optional[str]) -> str:
    if df.empty or df["mes_ano"].nunique() < 6:
        return "Para avaliar sazonalidade com confiança, preciso de pelo menos 6–12 meses de dados."
    # Mantém no máximo 12 meses mais recentes
    if df["mes_ano"].nunique() > 12:
        cutoff = df["mes_ano"].max() - pd.DateOffset(months=12)
        df = df[df["mes_ano"] > cutoff]

    # Tendência via inclinação de regressão linear simples
    idx = (df["mes_ano"].rank(method="dense").values - 1)  # 0..n-1
    y = df["valor"].values
    if len(idx) >= 2:
        x_mean, y_mean = idx.mean(), y.mean()
        num = ((idx - x_mean) * (y - y_mean)).sum()
        den = ((idx - x_mean) ** 2).sum() or 1.0
        slope = num / den
    else:
        slope = 0.0
    sinal = "alta" if slope > 0 else "queda" if slope < 0 else "estabilidade"

    # Top/bottom 3
    top = df.nlargest(3, "valor")[["mes_ano","valor"]]
    low = df.nsmallest(3, "valor")[["mes_ano","valor"]]

    def fmt_row(r):
        dt = pd.to_datetime(r["mes_ano"])
        mes_abbr = MONTH_NUM_TO_ABBR[int(dt.month)]
        val = f"R$ {r['valor']:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"{mes_abbr}/{int(dt.year)}: {val}"

    janela = f"{MONTH_NUM_TO_ABBR[int(df['mes'].min())]}/{int(df['ano'].min())} a {MONTH_NUM_TO_ABBR[int(df['mes'].max())]}/{int(df['ano'].max())}"
    parts = []
    if tipo_label:
        parts.append(f"**Indicador:** {tipo_label}")
    parts.append(f"**Janela analisada:** {janela}")
    parts.append(f"**Tendência geral (últimos {df['mes_ano'].nunique()} meses):** {sinal}.")
    parts.append("**Meses de pico (top 3):** " + "; ".join(fmt_row(r) for _, r in top.iterrows()))
    parts.append("**Meses de menor valor (bottom 3):** " + "; ".join(fmt_row(r) for _, r in low.iterrows()))
    return "\n".join(parts)

# ======== Variação ======== #
def _retrieve_with_filter(vector, question: str, vflt, k: int = 12):
    retriever = vector.as_retriever(search_kwargs={"k": k, "filter": vflt})
    try:
        return retriever.invoke(question)
    except Exception:
        # Vertex retriever às vezes usa .get_relevant_documents
        try:
            return retriever.get_relevant_documents(question)
        except Exception:
            return []

def retrieve_month_docs(vector, question: str, empresa_id: int, mes_abbr: str, ano: int, tipo: Optional[str], retries: int = 2):
    """Recupera docs para um mês/ano, com retry e relaxando 'tipo' se vier vazio."""
    # 1) com tipo
    vflt = build_vertex_filter(empresa_id, [mes_abbr], ano, tipo)
    for attempt in range(retries + 1):
        docs = _retrieve_with_filter(vector, question, vflt, k=12)
        if docs:
            return docs
        time.sleep(0.6 * (attempt + 1))  # backoff simples

    # 2) sem tipo (relax)
    vflt2 = build_vertex_filter(empresa_id, [mes_abbr], ano, None)
    for attempt in range(retries + 1):
        docs = _retrieve_with_filter(vector, question, vflt2, k=12)
        if docs:
            return docs
        time.sleep(0.6 * (attempt + 1))

    return []

def answer_variacao(vector, pergunta: str, empresa_id: int, meses_abbr: List[str], ano: int, tipo: Optional[str]):
    mes1, mes2 = meses_abbr[0], meses_abbr[1]
    docs1 = retrieve_month_docs(vector, pergunta, empresa_id, mes1, ano, tipo)
    docs2 = retrieve_month_docs(vector, pergunta, empresa_id, mes2, ano, tipo)

    if not docs1 or not docs2:
        faltou = []
        if not docs1: faltou.append(f"{mes1}/{ano}")
        if not docs2: faltou.append(f"{mes2}/{ano}")
        return {
            "answer": f"Não foi possível calcular a variação: faltam dados para {', '.join(faltou)}.",
            "source_documents": (docs1 or []) + (docs2 or [])
        }

    v1 = choose_record_value(docs1, tipo)
    v2 = choose_record_value(docs2, tipo)
    if v1 is None or v2 is None or v1 == 0:
        motivo = "valor não encontrado" if (v1 is None or v2 is None) else "valor do mês base é zero"
        return {"answer": f"Não foi possível calcular a variação ({motivo}).", "source_documents": docs1 + docs2}

    var = (v2 - v1) / v1 * 100.0
    sinal = "alta" if var > 0 else "queda" if var < 0 else "estabilidade"

    def br_money(x): return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    resp = (f"Faturamento em {mes1}/{ano}: {br_money(v1)}\n"
            f"Faturamento em {mes2}/{ano}: {br_money(v2)}\n"
            f"Variação: {var:.2f}% ({sinal}).")
    return {"answer": resp, "source_documents": docs1 + docs2}

# ======== Abrir Vector Search existente ======== #
def _load_ingest_state() -> Tuple[Optional[str], Optional[str]]:
    """
    Lê .ingest_state.json gerado pelo 01_ingest.py.
    Retorna (index_id, endpoint_id) ou (None, None) se não existir.
    """
    p = Path(".ingest_state.json")
    if not p.exists():
        return None, None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        return data.get("index_id"), data.get("endpoint_id")
    except Exception:
        return None, None

def _pick_latest_by_displayprefix(objs, name_prefix: str):
    """
    Entre objetos do SDK (Index/Endpoint), escolhe o mais recente cujo display_name
    seja exatamente name_prefix ou comece com name_prefix (cobre sufixo com timestamp).
    """
    filtered = []
    for o in objs:
        try:
            dn = o.display_name
        except Exception:
            try:
                dn = o.gca_resource.display_name
            except Exception:
                dn = ""
        if dn == name_prefix or (dn and dn.startswith(name_prefix)):
            filtered.append(o)
    if not filtered:
        return None
    # ordena por create_time (quando disponível)
    def _ctime(obj):
        try:
            ct = obj.gca_resource.create_time  # RFC3339
            return pd.to_datetime(ct)
        except Exception:
            return pd.Timestamp.min
    filtered.sort(key=_ctime, reverse=True)
    return filtered[0]

def open_vector_vertexai() -> VectorSearchVectorStore:
    if not PROJECT_ID or not BUCKET:
        raise RuntimeError("Defina GOOGLE_CLOUD_PROJECT e VERTEX_STAGING_BUCKET no ambiente.")
    aiplatform.init(project=PROJECT_ID, location=REGION, staging_bucket=BUCKET)

    # 1) Tenta abrir pelo estado salvo
    state_index_id, state_endpoint_id = (None, None) if IGNORE_STATE else _load_ingest_state()

    index_id = None
    endpoint_id = None

    try:
        if state_index_id:
            # Verifica se o index existe
            idx = aiplatform.MatchingEngineIndex(state_index_id)
            index_id = idx.resource_name.split("/")[-1]
    except Exception:
        index_id = None

    try:
        if state_endpoint_id:
            ep = aiplatform.MatchingEngineIndexEndpoint(state_endpoint_id)
            endpoint_id = ep.resource_name.split("/")[-1]
    except Exception:
        endpoint_id = None

    # 2) Fallback por "display_name" (pega o mais recente compatível com prefixo)
    if not index_id:
        all_idx = aiplatform.MatchingEngineIndex.list()
        picked = _pick_latest_by_displayprefix(all_idx, INDEX_NAME)
        if not picked:
            raise RuntimeError(f"Index com display_name '{INDEX_NAME}' (ou prefixo) não encontrado. Rode o 01_ingest.py antes.")
        index_id = picked.resource_name.split("/")[-1]

    if not endpoint_id:
        all_ep = aiplatform.MatchingEngineIndexEndpoint.list()
        picked_ep = _pick_latest_by_displayprefix(all_ep, ENDPOINT_NAME)
        if not picked_ep:
            raise RuntimeError(f"Endpoint com display_name '{ENDPOINT_NAME}' (ou prefixo) não encontrado. Rode o 01_ingest.py antes.")
        endpoint_id = picked_ep.resource_name.split("/")[-1]

        # Verifica se o endpoint escolhido tem o index implantado; caso não, tenta encontrar um endpoint que tenha.
        try:
            ep_obj = aiplatform.MatchingEngineIndexEndpoint(endpoint_id)
            deployed_ids = [d.index.split("/")[-1] for d in ep_obj.gca_resource.deployed_indexes]
            if index_id not in deployed_ids:
                # tenta achar outro endpoint com o index implantado
                for ep in all_ep:
                    try:
                        di = [d.index.split("/")[-1] for d in ep.gca_resource.deployed_indexes]
                        if index_id in di:
                            endpoint_id = ep.resource_name.split("/")[-1]
                            break
                    except Exception:
                        continue
                # Se ainda não achar, seguimos com o endpoint escolhido (consulta pode falhar)
                print("ℹ️  Aviso: o endpoint selecionado não tem o index escolhido implantado. "
                      "Se der erro de busca, reimplante via 01_ingest.py.")
        except Exception:
            pass

    # DEBUG: mostrar ambiente e recursos usados
    print(f"[DEBUG] GOOGLE_CLOUD_PROJECT={PROJECT_ID}")
    print(f"[DEBUG] REGION={REGION}  BUCKET={BUCKET}")
    print(f"[DEBUG] VERTEX_IGNORE_STATE={IGNORE_STATE}")
    print(f"[DEBUG] INDEX_NAME prefix={INDEX_NAME}  ENDPOINT_NAME prefix={ENDPOINT_NAME}")
    print(f"[DEBUG] Using Index ID: {index_id}")
    print(f"[DEBUG] Using Endpoint ID: {endpoint_id}")

    # Opcional: listar IDs deployados no endpoint
    try:
        ep_obj = aiplatform.MatchingEngineIndexEndpoint(endpoint_id)
        deployed_ids = [d.index.split("/")[-1] for d in ep_obj.gca_resource.deployed_indexes]
        print(f"[DEBUG] Deployed indexes on endpoint: {deployed_ids}")
    except Exception as _:
        pass

    embedding = VertexAIEmbeddings(model_name="text-embedding-005")  # 768 dims
    store = VectorSearchVectorStore.from_components(
        project_id=PROJECT_ID,
        region=REGION,
        gcs_bucket_name=BUCKET.replace("gs://", ""),
        index_id=index_id,
        endpoint_id=endpoint_id,
        embedding=embedding,
        stream_update=True,
    )
    return store

# ======== Fallback local (opcional) ======== #
def get_hf_embeddings():
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
    except Exception as e:
        raise RuntimeError("Fallback local indisponível: instale 'sentence-transformers' e 'langchain-huggingface'.") from e
    return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

def choose_loader(path: str) -> str:
    df = pd.read_excel(path, nrows=5)
    cols = {str(c).lower().strip() for c in df.columns}
    return "line" if {"mes_ano", "valor", "tipo", "empresa_id"}.issubset(cols) else "excel"

def _infer_tipo_from_context(context: str, sheet_name: str | None = None) -> Optional[str]:
    txt = f"{context or ''} || {sheet_name or ''}".lower()
    if "faturamento" in txt:
        return "faturamento"
    if "demanda" in txt:
        return "demanda"
    if "ticket" in txt or "tícket" in txt:
        return "ticket_medio"  # já normalizado
    return None

def excel_table_loader(path: str, empresa_id: int) -> List[Document]:
    dfs = pd.read_excel(path, sheet_name=None, header=None)
    docs: List[Document] = []
    MONTH_ALIAS = {"Jan":"Janeiro","Fev":"Fevereiro","Mar":"Março","Abr":"Abril","Mai":"Maio","Jun":"Junho",
                   "Jul":"Julho","Ago":"Agosto","Set":"Setembro","Out":"Outubro","Nov":"Novembro","Dez":"Dezembro"}
    for sheet_name, raw in dfs.items():
        if raw is None or raw.empty: continue
        first_row_idx = raw.first_valid_index()
        if first_row_idx is None: continue
        context = str(raw.iloc[first_row_idx, 0]).strip()
        header_idx = first_row_idx + 1
        header = [str(h).strip() for h in raw.iloc[header_idx]]
        data = raw.iloc[header_idx + 1:].copy()
        data.columns = header
        data = data.dropna(how="all")
        tipo_ctx = _infer_tipo_from_context(context, sheet_name)

        if "Mês" not in data.columns:
            meta = {"sheet": str(sheet_name), "empresa_id": str(empresa_id)}
            if tipo_ctx: meta["tipo"] = tipo_ctx
            docs.append(Document(f"{context}\n{data.to_csv(index=False, sep='|')}", metadata=meta))
            continue

        anos_cols = [c for c in data.columns if re.fullmatch(r"\d{4}", str(c))]
        long_df = pd.melt(data, id_vars=["Mês"], value_vars=anos_cols, var_name="Ano", value_name="Saldo").dropna(subset=["Saldo"])
        for _, row in long_df.iterrows():
            m_abrev = str(row["Mês"]).strip().replace(".", "")
            m_full  = MONTH_ALIAS.get(m_abrev, m_abrev)
            ano     = int(row["Ano"])
            saldo   = row["Saldo"]
            texto = f"Contexto: {context}\nMês: {m_full} ({m_abrev})\nAno: {ano}\nSaldo: {saldo}"
            meta = {"sheet": str(sheet_name), "mes": m_abrev, "ano": str(ano), "empresa_id": str(empresa_id)}
            if tipo_ctx: meta["tipo"] = tipo_ctx
            docs.append(Document(texto, metadata=meta))
    return docs

def line_per_value_loader(path: str) -> List[Document]:
    df = pd.read_excel(path)
    df.columns = [str(c).lower().strip() for c in df.columns]
    req = {"mes_ano", "valor", "tipo", "empresa_id"}
    if not req.issubset(set(df.columns)):
        raise ValueError("Colunas esperadas: mes_ano, valor, tipo, empresa_id.")
    docs: List[Document] = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row["mes_ano"])
        ano, mesN = int(dt.year), int(dt.month)
        mes_full, mes_abrev = MONTH_PT[mesN]
        valor = row["valor"]
        tipo  = normalize_tipo(row["tipo"])
        emp   = str(int(row["empresa_id"]))
        texto = (f"Empresa ID: {emp}\nTipo de indicador: {tipo}\nAno: {ano}\n"
                 f"Mês: {mes_full} ({mes_abrev})\nValor: {valor}")
        md: Dict[str, str] = {"empresa_id": emp, "ano": str(ano), "mes": mes_abrev}
        if tipo: md["tipo"] = tipo
        docs.append(Document(texto, metadata=md))
    return docs

def ensure_vector_hf(file_path: str, empresa_id: int):
    if not USE_LOCAL_FALLBACK:
        raise RuntimeError("Chroma não instalado. Instale 'langchain-chroma'.")
    from langchain_huggingface import HuggingFaceEmbeddings
    def get_hf_embeddings():
        return HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    stem = Path(file_path).stem
    store_dir = f"chromax_{stem}_hf"
    if os.path.exists(store_dir):
        return Chroma(persist_directory=store_dir, embedding_function=get_hf_embeddings())
    loader_kind = choose_loader(file_path)
    docs = line_per_value_loader(file_path) if loader_kind == "line" else excel_table_loader(file_path, empresa_id)
    return Chroma.from_documents(docs, embedding=get_hf_embeddings(), persist_directory=store_dir)

# ======== Cadeias ======== #
def make_chain_no_memory(retriever, llm):
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=None,
        combine_docs_chain_kwargs={"prompt": PROMPT_QA},
        return_source_documents=True,
    )

def validar_resposta(pergunta: str, provisoria: str, fontes_docs: List[Document]) -> str:
    fontes_txt = "\n".join(doc.page_content for doc in fontes_docs)
    hoje = datetime.now().strftime("%d/%m/%Y")
    n_sources = len(fontes_docs)
    validator = PROMPT_VALID | ChatVertexAI(model=GEMINI_VALID_MODEL, temperature=0.0, project=PROJECT_ID, location=REGION)
    try:
        feedback = validator.invoke(
            {"question": pergunta, "draft_answer": provisoria, "sources": fontes_txt, "hoje": hoje, "n_sources": str(n_sources)}
        ).content.strip()
        return provisoria if feedback == "PASS" else feedback
    except Exception:
        if USE_OPENAI_FALLBACK:
            try:
                validator2 = PROMPT_VALID | ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
                feedback = validator2.invoke(
                    {"question": pergunta, "draft_answer": provisoria, "sources": fontes_txt, "hoje": hoje, "n_sources": str(n_sources)}
                ).content.strip()
                return provisoria if feedback == "PASS" else feedback
            except Exception:
                pass
        return provisoria + "\n\n(Observação: validação automática indisponível no momento.)"

# ======== Invocação segura (captura 'not found in the storage') ======== #
DOCSTORE_MISSING_MSG = "not found in the storage"

def safe_invoke_chain(retriever, llm, question: str, memory: Optional[ConversationSummaryBufferMemory]):
    """
    Executa a chain e captura ValueError 'Documents ... not found in the storage'.
    Retorna (answer, fontes, err_docstore_mismatch: bool, raw_error: Optional[Exception])
    """
    chain = (ConversationalRetrievalChain.from_llm(
                llm=llm, retriever=retriever, memory=memory,
                combine_docs_chain_kwargs={"prompt": PROMPT_QA},
                return_source_documents=True
             ) if memory else make_chain_no_memory(retriever, llm))
    try:
        res = chain.invoke({"question": question} if memory else {"question": question, "chat_history": []})
        return res.get("answer", ""), res.get("source_documents", []) or [], False, None
    except ValueError as e:
        if DOCSTORE_MISSING_MSG in str(e).lower():
            # Caso clássico de índice ok e docstore GCS desincronizado.
            return "", [], True, e
        return "", [], False, e
    except Exception as e:
        return "", [], False, e

# ======== Execução principal (CLI) ======== #
def run_chat(empresa_id: int, local_fallback_file: Optional[str] = None, use_memory: bool = True):
    # 1) Abre Vector Search existente (Vertex) usando estado salvo ou o mais recente por prefixo
    store = open_vector_vertexai()

    # 2) LLM (Gemini) + memória (opcional)
    llm_qa = ChatVertexAI(model=GEMINI_QA_MODEL, temperature=0.5, project=PROJECT_ID, location=REGION)
    memory = ConversationSummaryBufferMemory(
        llm=ChatVertexAI(model=GEMINI_QA_MODEL, temperature=0.0, project=PROJECT_ID, location=REGION),
        memory_key="chat_history", output_key="answer", return_messages=True, max_token_limit=1500,
    ) if use_memory else None

    # 3) (Opcional) preparar fallback local
    vector_local = None
    if local_fallback_file:
        try:
            vector_local = ensure_vector_hf(local_fallback_file, empresa_id)
            print("ℹ️  Fallback local (Chroma+HF) ativado.")
        except Exception as e:
            print(f"⚠️  Não foi possível ativar fallback local: {e}")

    print("\n✅ Base pronta. Pergunte algo (ou 'sair').\n")
    while True:
        pergunta = input("Pergunta: ").strip()
        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            break

        tipo = parse_indicador(pergunta)
        tipo = normalize_tipo(tipo) if tipo else None  # garante normalização
        meses_abbr, ano = parse_meses_ano(pergunta)

        # ===== Intenção de série histórica / sazonalidade =====
        if is_time_series_intent(pergunta):
            docs = fetch_timeseries_docs(store, empresa_id, tipo, k=200)
            df = docs_to_timeseries(docs)
            if df.empty:
                print("\n🤖 Não encontrei uma série histórica suficiente para avaliar sazonalidade. "
                      "Tente especificar o indicador (ex.: 'sazonalidade do faturamento') ou verifique se há mais meses ingeridos.\n")
                continue
            rel = analyze_seasonality(df, tipo_label=tipo or "geral")
            print("\n🤖 " + rel + "\n")
            continue

        # Default mês atual se usuário não especificar
        assumiu_mes_atual = False
        if not meses_abbr:
            m_now, y_now = current_month_year()
            meses_abbr = [m_now]
            if ano is None: ano = y_now
            assumiu_mes_atual = True

        # Variação "mes1 e mes2 de ano"
        if len(meses_abbr) == 2 and ano is not None:
            try:
                res = answer_variacao(store, pergunta, empresa_id, meses_abbr, ano, tipo)
                answer, fontes = res["answer"], res.get("source_documents", []) or []
                if fontes:
                    final = validar_resposta(pergunta, answer, fontes)
                    print("\n🤖", final, "\n")
                    continue
            except Exception as e:
                print(f"\n🤖 Erro ao calcular variação: {e}\n")

        # 1ª tentativa: filtro com tipo (se houver)
        vertex_flt = build_vertex_filter(empresa_id, meses_abbr, ano, tipo)
        retriever = store.as_retriever(search_kwargs={"k": 8, "filter": vertex_flt})
        answer, fontes, docstore_mismatch, err = safe_invoke_chain(retriever, llm_qa, pergunta, memory)

        # Se deu docstore mismatch → tenta fallback local (se disponível)
        if docstore_mismatch:
            print("⚠️  Vertex retornou IDs sem payload no docstore. Tentando fallback local (se disponível).")
            if vector_local:
                try:
                    res_local = ConversationalRetrievalChain.from_llm(
                        llm=llm_qa,
                        retriever=vector_local.as_retriever(search_kwargs={"k": 8}),
                        memory=None,
                        combine_docs_chain_kwargs={"prompt": PROMPT_QA},
                        return_source_documents=True,
                    ).invoke({"question": pergunta, "chat_history": []})
                    answer = (res_local.get("answer","") + "\n\n(Nota: fallback local ativado por docstore fora de sincronia.)")
                    fontes = res_local.get("source_documents", []) or []
                except Exception as e:
                    print(f"\n🤖 Fallback local falhou: {e}\n")
            else:
                print("ℹ️  Passe --local-fallback-file para ativar Chroma+HF quando ocorrer esse cenário.")
        elif err:
            # Outro erro qualquer
            print(f"\n🤖 Erro na consulta: {err}\n")

        # Relaxa 'tipo' se não veio nada e não houve docstore mismatch
        if not fontes and not docstore_mismatch and tipo:
            vertex_flt_no_tipo = build_vertex_filter(empresa_id, meses_abbr, ano, None)
            retriever2 = store.as_retriever(search_kwargs={"k": 8, "filter": vertex_flt_no_tipo})
            answer, fontes, docstore_mismatch, err = safe_invoke_chain(retriever2, llm_qa, pergunta, None)

            if docstore_mismatch and vector_local:
                print("⚠️  Docstore fora de sincronia nessa busca sem 'tipo'. Usando fallback local.")
                try:
                    res_local = ConversationalRetrievalChain.from_llm(
                        llm=llm_qa,
                        retriever=vector_local.as_retriever(search_kwargs={"k": 8}),
                        memory=None,
                        combine_docs_chain_kwargs={"prompt": PROMPT_QA},
                        return_source_documents=True,
                    ).invoke({"question": pergunta, "chat_history": []})
                    answer = (res_local.get("answer","") + "\n\n(Nota: fallback local ativado por docstore fora de sincronia.)")
                    fontes = res_local.get("source_documents", []) or []
                except Exception as e:
                    print(f"\n🤖 Fallback local falhou: {e}\n")

        # Recuo mês a mês (se assumiu mês atual e ainda não achou nada)
        if not fontes and assumiu_mes_atual:
            m_try, y_try = meses_abbr[0], ano
            for _ in range(24):
                m_try, y_try = previous_month(m_try, y_try)
                vertex_fb = build_vertex_filter(empresa_id, [m_try], y_try, tipo)
                retr_fb = store.as_retriever(search_kwargs={"k": 8, "filter": vertex_fb})
                ans_fb, fontes_fb, mismatch_fb, err_fb = safe_invoke_chain(retr_fb, llm_qa, pergunta, None)
                if mismatch_fb:
                    print(f"⚠️  Docstore fora de sincronia ao buscar {m_try}/{y_try}.")
                    if vector_local:
                        try:
                            res_local = ConversationalRetrievalChain.from_llm(
                                llm=llm_qa,
                                retriever=vector_local.as_retriever(search_kwargs={"k": 8}),
                                memory=None,
                                combine_docs_chain_kwargs={"prompt": PROMPT_QA},
                                return_source_documents=True,
                            ).invoke({"question": pergunta, "chat_history": []})
                            answer = (res_local.get("answer","") + f"\n\n(Nota: fallback local ativado; Vertex retornou IDs sem payload em {m_try}/{y_try}.)")
                            fontes = res_local.get("source_documents", []) or []
                            break
                        except Exception as e:
                            print(f"\n🤖 Fallback local falhou: {e}\n")
                    continue

                if fontes_fb:
                    info = f"(Sem dados para {meses_abbr[0]}/{ano}. Mostrando o último disponível: {m_try}/{y_try}.)"
                    answer = ans_fb
                    fontes = fontes_fb
                    pergunta = f"{pergunta}\n{info}"
                    break

        # Se ainda sem fontes → mensagem objetiva
        if not fontes:
            print("\n🤖 Não encontrei dados compatíveis com os filtros "
                  f"(empresa_id={empresa_id}"
                  f"{', meses='+','.join(meses_abbr) if meses_abbr else ''}"
                  f"{', ano='+str(ano) if ano else ''}"
                  f"{', tipo='+tipo if tipo else ''}).\n")
            continue

        final = validar_resposta(pergunta, answer, fontes)
        print("\n🤖", final, "\n")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="RAG para planilhas de indicadores financeiros (serving, sem ingestão)")
    ap.add_argument("--empresa", required=True, type=int, help="empresa_id (inteiro)")
    ap.add_argument("--local-fallback-file", default=None,
                    help="(Opcional) Excel para fallback local (Chroma+HF) quando Vertex/LLM indisponíveis ou docstore fora de sincronia")
    ap.add_argument("--no-memory", action="store_true", help="Desativa memória de conversa")
    args = ap.parse_args()

    if not PROJECT_ID or not BUCKET:
        raise RuntimeError("Defina GOOGLE_CLOUD_PROJECT e VERTEX_STAGING_BUCKET no ambiente ou .env.")

    run_chat(empresa_id=args.empresa,
             local_fallback_file=args.local_fallback_file,
             use_memory=not args.no_memory)
