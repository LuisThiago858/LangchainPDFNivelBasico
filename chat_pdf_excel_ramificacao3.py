"""
RAG para planilhas de indicadores financeiros
(com memória + filtro dinâmico por mês/ano + validação anti-alucinação
 + default para mês atual com fallback para último disponível + filtro por indicador)

Requisitos:
    pip install -U langchain langchain-openai langchain-chroma python-dotenv pandas
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

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate

load_dotenv()

# ========= 1. Config ========= #
BASES = {
    "4": "indicadores_financeiros_almater.xlsx",
    "5": "indicadores_base_2.xlsx",
}

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

# LLMs (temperatura 0 para reduzir variação)
LLM_QA   = ChatOpenAI(model="gpt-4o-mini", temperature=0)
LLM_VALID = ChatOpenAI(model="gpt-4o-mini", temperature=0)
# LCEL (remove deprecação de LLMChain)
VALIDATOR = PROMPT_VALID | LLM_VALID

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

# nomes/abreviações aceitas -> abreviação canônica
MONTH_NAME_TO_ABBR = {
    "janeiro": "Jan", "fevereiro": "Fev", "março": "Mar", "marco": "Mar",
    "abril": "Abr", "maio": "Mai", "junho": "Jun", "julho": "Jul",
    "agosto": "Ago", "setembro": "Set", "outubro": "Out",
    "novembro": "Nov", "dezembro": "Dez",
    "jan": "Jan","fev":"Fev","mar":"Mar","abr":"Abr","mai":"Mai","jun":"Jun",
    "jul":"Jul","ago":"Ago","set":"Set","out":"Out","nov":"Nov","dez":"Dez",
}

# ========= 2A. Loader – tabela cruzada (inclui empresa_id e tenta inferir 'tipo') ========= #
def _infer_tipo_from_context(context: str) -> Optional[str]:
    s = (context or "").lower()
    if "faturamento" in s:
        return "faturamento"
    if "demanda" in s:
        return "demanda"
    if "ticket" in s or "tícket" in s:
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

        tipo_ctx = _infer_tipo_from_context(context)

        if "Mês" not in data.columns:
            docs.append(
                Document(
                    f"{context}\n{data.to_csv(index=False, sep='|')}",
                    metadata={"sheet": sheet_name, "empresa_id": empresa_id, **({"tipo": tipo_ctx} if tipo_ctx else {})}
                )
            )
            continue

        anos_cols = [c for c in data.columns if re.fullmatch(r"\d{4}", str(c))]
        long_df = pd.melt(
            data, id_vars=["Mês"], value_vars=anos_cols,
            var_name="Ano", value_name="Saldo"
        ).dropna(subset=["Saldo"])

        for _, row in long_df.iterrows():
            m_abrev = str(row["Mês"]).strip()
            m_full  = MONTH_ALIAS.get(m_abrev, m_abrev)
            ano, saldo = int(row["Ano"]), row["Saldo"]

            texto = (
                f"Contexto: {context}\n"
                f"Mês: {m_full} ({m_abrev})\n"
                f"Ano: {ano}\n"
                f"Saldo: {saldo}"
            )
            meta = {
                "sheet": sheet_name,
                "mes": m_abrev, "ano": ano,
                "empresa_id": empresa_id
            }
            if tipo_ctx:
                meta["tipo"] = tipo_ctx
            docs.append(Document(texto, metadata=meta))
    return docs

# ========= 2B. Loader – linha a linha ========= #
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
        emp = int(row["empresa_id"])
        texto = (
            f"Empresa ID: {emp}\n"
            f"Tipo de indicador: {tipo}\n"
            f"Ano: {ano}\n"
            f"Mês: {mes_full} ({mes_abrev})\n"
            f"Valor: {valor}"
        )
        docs.append(Document(texto, metadata={
            "empresa_id": emp, "tipo": tipo, "ano": ano, "mes": mes_abrev
        }))
    return docs

# ========= 2C. Escolha loader ========= #
def choose_loader(path: str) -> str:
    df = pd.read_excel(path, nrows=5)
    cols = {c.lower() for c in df.columns}
    return "line" if {"mes_ano", "valor", "tipo", "empresa_id"}.issubset(cols) else "cross"

# ========= 3. Util: parse indicador e mês/ano na pergunta ========= #
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
    Suporta 1 ou 2 meses (junho e maio de 2025, etc.).
    """
    s = pergunta.lower()

    # mes1 e mes2 de 2025
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

    # mes1 de 2025 e mes2 de 2025
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

    # abbr1/2025 e abbr2/2025
    m_pair_abbr = re.findall(r"(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[-/](\d{4})", s)
    if len(m_pair_abbr) >= 2:
        (m1_txt, a1), (m2_txt, a2) = m_pair_abbr[0], m_pair_abbr[1]
        if a1 == a2:
            m1 = MONTH_NAME_TO_ABBR.get(m1_txt); m2 = MONTH_NAME_TO_ABBR.get(m2_txt)
            meses = [m for m in (m1, m2) if m]
            return (meses, int(a1)) if meses else ([], None)

    # 06/2025 e 05/2025
    m_pair_num = re.findall(r"\b(1[0-2]|0?[1-9])[/-](\d{4})\b", s)
    if len(m_pair_num) >= 2:
        (m1n, a1), (m2n, a2) = m_pair_num[0], m_pair_num[1]
        if a1 == a2:
            _, m1 = MONTH_PT[int(m1n)]; _, m2 = MONTH_PT[int(m2n)]
            return ([m1, m2], int(a1))

    # único mês/ano
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

# ========= 4. Filtro Chroma ($and + $or para meses múltiplos) ========= #
def build_chroma_filter(empresa_id: int, meses_abbr: List[str], ano: Optional[int], tipo: Optional[str]):
    clauses = [{"empresa_id": {"$eq": int(empresa_id)}}]
    if ano is not None:
        clauses.append({"ano": {"$eq": int(ano)}})
    if tipo:
        clauses.append({"tipo": {"$eq": tipo}})

    if meses_abbr:
        if len(meses_abbr) == 1:
            clauses.append({"mes": {"$eq": meses_abbr[0]}})
        else:
            clauses.append({"$or": [{"mes": {"$eq": m}} for m in meses_abbr]})

    return clauses[0] if len(clauses) == 1 else {"$and": clauses}

# ========= 5. Vetor + memória ========= #
def build_vector_and_memory(file_path: str, empresa_id: int):
    store_dir = f"chromax_{Path(file_path).stem}"
    loader_kind = choose_loader(file_path)

    if os.path.exists(store_dir):
        vector = Chroma(persist_directory=store_dir, embedding_function=OpenAIEmbeddings())
    else:
        if loader_kind == "line":
            docs = line_per_value_loader(file_path)
        else:
            docs = excel_table_loader(file_path, empresa_id=empresa_id)
        vector = Chroma.from_documents(docs, embedding=OpenAIEmbeddings(), persist_directory=store_dir)

    memory = ConversationSummaryBufferMemory(
        llm=LLM_QA,
        memory_key="chat_history",
        output_key="answer",
        return_messages=True,
        max_token_limit=1500,
    )
    return vector, memory

def make_chain(retriever, memory):
    return ConversationalRetrievalChain.from_llm(
        llm=LLM_QA,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT_QA},
        return_source_documents=True,
    )

# ========= 6. Validação ========= #
def validar_resposta(pergunta: str, provisoria: str, fontes_docs: List[Document]) -> str:
    fontes_txt = "\n".join(doc.page_content for doc in fontes_docs)
    hoje = datetime.now().strftime("%d/%m/%Y")
    n_sources = len(fontes_docs)
    feedback = VALIDATOR.invoke(
        {"question": pergunta, "draft_answer": provisoria, "sources": fontes_txt, "hoje": hoje, "n_sources": str(n_sources)}
    ).content.strip()
    return provisoria if feedback == "PASS" else feedback

# ========= 7. Fallback para mês atual / último disponível ========= #
def current_month_year():
    now = datetime.now()
    return MONTH_PT[now.month][1], now.year  # ("Jul", 2025) por ex.

def previous_month(m_abbr: str, ano: int):
    m_num = MONTH_ABBR_TO_NUM[m_abbr]
    m_num -= 1
    if m_num == 0:
        return "Dez", ano - 1
    return MONTH_NUM_TO_ABBR[m_num], ano

def answer_with_fallback(vector, memory, pergunta: str, empresa_id: int, tipo: Optional[str], meses_abbr: List[str], ano: Optional[int]):
    """
    Se não houver mês na pergunta: assume mês atual; se não achar fonte, recua mês a mês (até 24) e informa isso.
    """
    assumiu_mes_atual = False
    if not meses_abbr:
        m_now, y_now = current_month_year()
        meses_abbr = [m_now]
        ano = y_now if ano is None else ano
        assumiu_mes_atual = True

    # tenta 1ª vez (ou com 2 meses, para variação)
    chroma_filter = build_chroma_filter(empresa_id, meses_abbr, ano, tipo)
    retriever = vector.as_retriever(search_kwargs={"k": 8, "filter": chroma_filter})
    qa = make_chain(retriever, memory)
    res = qa.invoke({"question": pergunta})
    fontes = res.get("source_documents", []) or []

    if fontes or not assumiu_mes_atual or len(meses_abbr) > 1:
        return res, None  # sem fallback ou com variação (2 meses)

    # fallback: recua até achar
    m_try, y_try = meses_abbr[0], ano
    for _ in range(24):  # recua até 2 anos
        m_try, y_try = previous_month(m_try, y_try)
        chroma_filter_fb = build_chroma_filter(empresa_id, [m_try], y_try, tipo)
        retriever_fb = vector.as_retriever(search_kwargs={"k": 8, "filter": chroma_filter_fb})
        qa_fb = make_chain(retriever_fb, memory)
        res_fb = qa_fb.invoke({"question": pergunta})
        fontes_fb = res_fb.get("source_documents", []) or []
        if fontes_fb:
            info = f"(Sem dados para {meses_abbr[0]}/{ano}. Mostrando o último disponível: {m_try}/{y_try}.)"
            return res_fb, info

    # nada encontrado
    return {"answer": "", "source_documents": []}, "Não encontrei dados para o mês atual nem para meses anteriores (até 24 meses)."

# ========= 8. CLI ========= #
if __name__ == "__main__":
    file_path = BASES["5"]  # troque para "4" se quiser
    # Se mudar metadados/tipos, reindexe:
    # shutil.rmtree(f"chromax_{Path(file_path).stem}", ignore_errors=True)

    empresa_id = int(input("empresa_id para esta sessão: ").strip())
    vector, memory = build_vector_and_memory(file_path, empresa_id)
    print("✅ Base carregada!\n")

    while True:
        pergunta = input("Pergunta (ou 'sair'): ").strip()
        if pergunta.lower() == "sair":
            break

        # --- parse indicador + meses/ano ---
        tipo = parse_indicador(pergunta)  # "faturamento" | "demanda" | "ticket médio" | None
        meses_abbr, ano = parse_meses_ano(pergunta)   # lista (0, 1 ou 2 meses) + ano

        # ---------- 1ª passada (com fallback se necessário) ----------
        res, fallback_info = answer_with_fallback(vector, memory, pergunta, empresa_id, tipo, meses_abbr, ano)
        provisoria = res["answer"]
        fontes = res.get("source_documents", []) or []

        # ---------- Falta de fontes -> resposta segura ----------
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

        # ---------- 2ª passada (validação) ----------
        final = validar_resposta(
            pergunta if not fallback_info else f"{pergunta}\n{fallback_info}",
            provisoria,
            fontes
        )
        print("\n🤖", final, "\n")

