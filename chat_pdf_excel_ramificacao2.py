"""
RAG para planilhas de indicadores financeiros
(com memória + filtro dinâmico por mês/ano + validação anti-alucinação)

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
from langchain.chains import LLMChain, ConversationalRetrievalChain
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
        "Você é a SDR da AG Consultoria, chamada Gisa. Você é uma Analista de Finanças experiente e muito inteligente. Sua abordagem é sempre amigável e cordial, porém séria e confiável. Você precisa transmitir conhecimento e segurança ao explicar os indicadores financeiros dos clientes. Use os trechos abaixo para responder com exatidão.\n"
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
LLM_QA = ChatOpenAI(model="gpt-4o-mini", temperature=0)
LLM_VALID = ChatOpenAI(model="gpt-4o-mini", temperature=0)
validator_chain = LLMChain(llm=LLM_VALID, prompt=PROMPT_VALID)

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
# nomes/abreviações aceitas -> abreviação canônica
MONTH_NAME_TO_ABBR = {
    "janeiro": "Jan", "fevereiro": "Fev", "março": "Mar", "marco": "Mar",
    "abril": "Abr", "maio": "Mai", "junho": "Jun", "julho": "Jul",
    "agosto": "Ago", "setembro": "Set", "outubro": "Out",
    "novembro": "Nov", "dezembro": "Dez",
    "jan": "Jan","fev":"Fev","mar":"Mar","abr":"Abr","mai":"Mai","jun":"Jun",
    "jul":"Jul","ago":"Ago","set":"Set","out":"Out","nov":"Nov","dez":"Dez",
}

# ========= 2A. Loader – tabela cruzada (inclui empresa_id) ========= #
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

        if "Mês" not in data.columns:
            docs.append(
                Document(
                    f"{context}\n{data.to_csv(index=False, sep='|')}",
                    metadata={"sheet": sheet_name, "empresa_id": empresa_id}
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
            docs.append(
                Document(
                    texto,
                    metadata={
                        "sheet": sheet_name,
                        "mes": m_abrev, "ano": ano,
                        "empresa_id": empresa_id
                    }
                )
            )
    return docs

# ========= 2B. Loader – linha a linha ========= #
def line_per_value_loader(path: str) -> List[Document]:
    df = pd.read_excel(path)
    # padroniza nomes para lower
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
        tipo = str(row["tipo"]).strip()
        emp = int(row["empresa_id"])
        texto = (
            f"Empresa ID: {emp}\n"
            f"Tipo de indicador: {tipo}\n"
            f"Ano: {ano}\n"
            f"Mês: {mes_full} ({mes_abrev})\n"
            f"Valor: {valor}"
        )
        docs.append(
            Document(
                texto,
                metadata={"empresa_id": emp, "tipo": tipo, "ano": ano, "mes": mes_abrev}
            )
        )
    return docs

# ========= 2C. Escolha loader ========= #
def choose_loader(path: str) -> str:
    df = pd.read_excel(path, nrows=5)
    cols = {c.lower() for c in df.columns}
    return "line" if {"mes_ano", "valor", "tipo", "empresa_id"}.issubset(cols) else "cross"

# ========= 3. Util: parse mês/ano na pergunta ========= #

def parse_meses_ano(pergunta: str):
    """
    Retorna (lista_meses_abrev, ano_int) quando encontrado; caso contrário ([], None).
    Aceita:
      - 'junho e maio de 2025'
      - 'junho de 2025 e maio de 2025'
      - 'jun/2025 e mai/2025'
      - '06/2025 e 05/2025'
      - também funciona com 1 mês apenas.
    """
    s = pergunta.lower()

    # Caso 'mes1 e mes2 de 2025'
    m_pair = re.search(
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*e\s*"
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*(?:de)?\s*(\d{4})",
        s
    )
    if m_pair:
        mes1_txt, mes2_txt, ano = m_pair.group(1), m_pair.group(2), int(m_pair.group(3))
        m1 = MONTH_NAME_TO_ABBR.get(mes1_txt)
        m2 = MONTH_NAME_TO_ABBR.get(mes2_txt)
        meses = [m for m in (m1, m2) if m]
        return (meses, ano) if meses else ([], None)

    # Caso 'mes1 de 2025 e mes2 de 2025'
    m_pair_dup = re.search(
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*(?:de)?\s*(\d{4})\s*e\s*"
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*(?:de)?\s*(\d{4})",
        s
    )
    if m_pair_dup:
        mes1_txt, ano1, mes2_txt, ano2 = m_pair_dup.group(1), int(m_pair_dup.group(2)), m_pair_dup.group(3), int(m_pair_dup.group(4))
        if ano1 == ano2:
            m1 = MONTH_NAME_TO_ABBR.get(mes1_txt)
            m2 = MONTH_NAME_TO_ABBR.get(mes2_txt)
            meses = [m for m in (m1, m2) if m]
            return (meses, ano1) if meses else ([], None)

    # Caso 'jun/2025 e mai/2025'
    m_pair_abbr = re.findall(r"(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[-/](\d{4})", s)
    if len(m_pair_abbr) >= 2:
        # Considere o mesmo ano para os dois, se iguais
        (m1_txt, a1), (m2_txt, a2) = m_pair_abbr[0], m_pair_abbr[1]
        if a1 == a2:
            m1 = MONTH_NAME_TO_ABBR.get(m1_txt)
            m2 = MONTH_NAME_TO_ABBR.get(m2_txt)
            meses = [m for m in (m1, m2) if m]
            return (meses, int(a1)) if meses else ([], None)

    # Caso '06/2025 e 05/2025'
    m_pair_num = re.findall(r"\b(1[0-2]|0?[1-9])[/-](\d{4})\b", s)
    if len(m_pair_num) >= 2:
        (m1n, a1), (m2n, a2) = m_pair_num[0], m_pair_num[1]
        if a1 == a2:
            _, m1 = MONTH_PT[int(m1n)]
            _, m2 = MONTH_PT[int(m2n)]
            return ([m1, m2], int(a1))

    # Fallback: tentar um único mês/ano
    # 'junho de 2025' / 'jun/2025' / '06/2025'
    # (reaproveita sua lógica anterior)
    # 1) nome + ano
    m1 = re.search(
        r"(jan(?:eiro)?|fev(?:ereiro)?|mar(?:ço|co)?|abr(?:il)?|mai(?:o)?|jun(?:ho)?|jul(?:ho)?|ago(?:sto)?|set(?:embro)?|out(?:ubro)?|nov(?:embro)?|dez(?:embro)?)\s*(?:de)?\s*(\d{4})",
        s
    )
    if m1:
        mes_txt, ano = m1.group(1), int(m1.group(2))
        m = MONTH_NAME_TO_ABBR.get(mes_txt)
        return ([m], ano) if m else ([], None)

    # 2) abbr + ano
    m2 = re.search(r"(jan|fev|mar|abr|mai|jun|jul|ago|set|out|nov|dez)[-/](\d{4})", s)
    if m2:
        m = MONTH_NAME_TO_ABBR.get(m2.group(1))
        return ([m], int(m2.group(2))) if m else ([], None)

    # 3) num + ano
    m3 = re.search(r"\b(1[0-2]|0?[1-9])[/-](\d{4})\b", s)
    if m3:
        _, m = MONTH_PT[int(m3.group(1))]
        return ([m], int(m3.group(2)))

    return ([], None)

# ========= 4. Filtro Chroma ($and) ========= #
def build_chroma_filter(empresa_id: int, meses_abbr: list, ano: int | None):
    clauses = [{"empresa_id": {"$eq": int(empresa_id)}}]
    if ano is not None:
        clauses.append({"ano": {"$eq": int(ano)}})

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
    feedback = validator_chain.predict(
        question=pergunta,
        draft_answer=provisoria,
        sources=fontes_txt,
        hoje=hoje,
        n_sources=str(n_sources),
    ).strip()
    return provisoria if feedback == "PASS" else feedback

# ========= 7. CLI ========= #
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

        # --- filtros dinâmicos com parse da pergunta ---
        meses_abbr, ano = parse_meses_ano(pergunta)   # <- agora retorna lista
        chroma_filter = build_chroma_filter(empresa_id, meses_abbr, ano)

        retriever = vector.as_retriever(
            search_kwargs={"k": 8, "filter": chroma_filter}
        )
        qa = make_chain(retriever, memory)

        # ---------- 1ª passada ----------
        res = qa.invoke({"question": pergunta})
        provisoria = res["answer"]
        fontes = res.get("source_documents", []) or []

        # ---------- Falta de fontes -> resposta segura ----------
        if len(fontes) == 0:
            msg = (
                f"Não encontrei dados compatíveis com os filtros "
                f"(empresa_id={empresa_id}"
                f"{', mês='+mes_abbr if mes_abbr else ''}"
                f"{', ano='+str(ano) if ano else ''}). "
                "Verifique se o arquivo contém esse período/empresa."
            )
            print("\n🤖 " + msg + "\n")
            continue

        # ---------- 2ª passada (validação) ----------
        final = validar_resposta(pergunta, provisoria, fontes)
        print("\n🤖", final, "\n")
