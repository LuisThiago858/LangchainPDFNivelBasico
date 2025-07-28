"""
RAG para planilhas de indicadores financeiros
(com memória automática de conversa)

Requisitos:
    pip install langchain langchain-openai langchain-chroma python-dotenv pandas
"""

# ========= 0. Imports ========= #
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryBufferMemory
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from pathlib import Path
import pandas as pd
import re, os, shutil

load_dotenv()

# ========= 1. Config ========= #
BASES = {
    "4": "indicadores_financeiros_almater.xlsx",
    "5": "indicadores_base_2.xlsx",
}

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Você e um analista financeiro. Use os trechos abaixo para responder com exatidão.\n"
        "Se a resposta estiver em tabela, extraia o valor direto.\n\n"
        "Pergunta: {question}\n"
        "========\n{context}\n========\nResposta:"
    ),
)

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

# ========= 2A. Loader – tabela cruzada ========= #
def excel_table_loader(path: str) -> list[Document]:
    dfs = pd.read_excel(path, sheet_name=None, header=None)
    docs: list[Document] = []
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
            docs.append(Document(f"{context}\n{data.to_csv(index=False, sep='|')}",
                                 metadata={"sheet": sheet_name}))
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
            docs.append(Document(texto, metadata={"sheet": sheet_name,
                                                  "mes": m_abrev, "ano": ano}))
    return docs

# ========= 2B. Loader – linha a linha ========= #
def line_per_value_loader(path: str) -> list[Document]:
    df = pd.read_excel(path)
    req = {"mes_ano", "valor", "tipo", "empresa_id"}
    if not req.issubset(df.columns.str.lower()):
        raise ValueError("Colunas esperadas não encontradas.")
    docs = []
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
        docs.append(Document(texto, metadata={"empresa_id": emp, "tipo": tipo,
                                              "ano": ano, "mes": mes_abrev}))
    return docs

# ========= 2C. Escolha loader ========= #
def choose_loader(path: str):
    df = pd.read_excel(path, nrows=5)
    cols = {c.lower() for c in df.columns}
    return line_per_value_loader if {"mes_ano", "valor", "tipo", "empresa_id"}.issubset(cols) else excel_table_loader

# ========= 3. Builder ========= #
def build_qa(file_path: str) -> ConversationalRetrievalChain:
    store_dir = f"chromax_{Path(file_path).stem}"
    if os.path.exists(store_dir):
        vector = Chroma(persist_directory=store_dir,
                        embedding_function=OpenAIEmbeddings())
    else:
        docs = choose_loader(file_path)(file_path)
        vector = Chroma.from_documents(
            docs, embedding=OpenAIEmbeddings(), persist_directory=store_dir
        )

    memory = ConversationSummaryBufferMemory(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        memory_key="chat_history",      # nome esperado pela chain
        output_key="answer",            # <<<<<< CHAVE IMPORTANTE
        return_messages=True,
        max_token_limit=1500,
    )

     # ⚠️ Aqui garantimos o filtro por empresa_id no nível do retriever
    retriever = vector.as_retriever(
        search_kwargs={"k": 8, "filter": {"empresa_id": empresa_id}}
    )

    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=retriever,
        #vector.as_retriever(search_kwargs={"k": 8}),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=True,
    )

# ========= 4. CLI ========= #
if __name__ == "__main__":
    file_path = BASES["5"]  # troque para "4" se quiser
    # shutil.rmtree(f"chromax_{Path(file_path).stem}", ignore_errors=True)

    empresa_id = int(input("empresa_id para esta sessão: ").strip())

    qa = build_qa(file_path)
    print("✅ Base carregada!\n")

    while True:
        pergunta = input("Pergunta (ou 'sair'): ").strip()
        if pergunta.lower() == "sair":
            break

        res = qa.invoke({"question": pergunta})
        print("\n🤖", res["answer"], "\n")
