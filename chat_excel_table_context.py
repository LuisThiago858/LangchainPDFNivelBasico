"""
RAG para planilhas de indicadores financeiros
(com memória de conversa)
"""

# ========= 0. Imports ========= #
from langchain_chroma import Chroma                     # <- pacote novo
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

from pathlib import Path
import pandas as pd
import re, os, shutil

load_dotenv()   # carrega OPENAI_API_KEY, etc.

# ========= 1. Configs ========= #
BASES = {
    "4": "indicadores_financeiros_almater.xlsx",   # tabela mês x ano
    "5": "indicadores_base_2.xlsx",                # linha-a-linha
}

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Use os trechos abaixo para responder com exatidão.\n"
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

        data = raw.iloc[header_idx + 1:].copy()
        data.columns = header
        data = data.dropna(how="all")

        if "Mês" not in data.columns:
            texto = data.to_csv(index=False, sep="|")
            docs.append(Document(page_content=f"{context}\n{texto}",
                                 metadata={"sheet": sheet_name}))
            continue

        id_vars = ["Mês"]
        anos_cols = [c for c in data.columns if re.fullmatch(r"\d{4}", str(c))]
        long_df = pd.melt(
            data, id_vars=id_vars, value_vars=anos_cols,
            var_name="Ano", value_name="Saldo"
        ).dropna(subset=["Saldo"])

        for _, row in long_df.iterrows():
            mes_abrev = str(row["Mês"]).strip()
            mes_full  = MONTH_ALIAS.get(mes_abrev, mes_abrev)
            ano, saldo = int(row["Ano"]), row["Saldo"]

            texto = (
                f"Contexto: {context}\n"
                f"Mês: {mes_full} ({mes_abrev})\n"
                f"Ano: {ano}\n"
                f"Saldo: {saldo}"
            )
            meta = {"sheet": sheet_name, "mes": mes_abrev, "ano": ano}
            docs.append(Document(page_content=texto, metadata=meta))
    return docs

# ========= 2B. Loader – linha a linha ========= #
def line_per_value_loader(path: str) -> list[Document]:
    df = pd.read_excel(path)
    required = {"mes_ano", "valor", "tipo", "empresa_id"}
    if not required.issubset(df.columns.str.lower()):
        raise ValueError("Colunas esperadas não encontradas.")

    docs: list[Document] = []
    for _, row in df.iterrows():
        dt = pd.to_datetime(row["mes_ano"])
        ano, mesN = int(dt.year), int(dt.month)
        mes_full, mes_abrev = MONTH_PT[mesN]

        valor = row["valor"]
        tipo = str(row["tipo"]).strip()
        empresa_id = int(row["empresa_id"])

        texto = (
            f"Empresa ID: {empresa_id}\n"
            f"Tipo de indicador: {tipo}\n"
            f"Ano: {ano}\n"
            f"Mês: {mes_full} ({mes_abrev})\n"
            f"Valor: {valor}"
        )
        meta = {"empresa_id": empresa_id, "tipo": tipo, "ano": ano, "mes": mes_abrev}
        docs.append(Document(page_content=texto, metadata=meta))
    return docs

# ========= 2C. Escolha automática ========= #
def choose_loader(path: str):
    df_sample = pd.read_excel(path, nrows=5)
    cols = {c.lower() for c in df_sample.columns}
    if {"mes_ano", "valor", "tipo", "empresa_id"}.issubset(cols):
        return line_per_value_loader
    return excel_table_loader     # fallback seguro

# ========= 3. Build QA ========= #
def build_qa(file_path: str) -> ConversationalRetrievalChain:
    store_dir = f"chromax_{Path(file_path).stem}"

    if os.path.exists(store_dir):
        vector = Chroma(persist_directory=store_dir,
                        embedding_function=OpenAIEmbeddings())
    else:
        loader = choose_loader(file_path)
        docs = loader(file_path)
        vector = Chroma.from_documents(
            docs, embedding=OpenAIEmbeddings(), persist_directory=store_dir
        )

    retriever = vector.as_retriever(search_kwargs={"k": 8})
    return ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=retriever,
        # era qa_prompt=…  / era chain_type_kwargs=…
        combine_docs_chain_kwargs={"prompt": PROMPT},   # <- ESTE é o nome aceito
        return_source_documents=True,
    )

# ========= 4. CLI ========= #
if __name__ == "__main__":
    file_path = BASES["5"]                    # troque para "4" se quiser

    # Se a planilha mudou, recrie o índice:
    # shutil.rmtree(f"chromax_{Path(file_path).stem}", ignore_errors=True)

    qa = build_qa(file_path)
    print("✅ Base carregada!\n")

    chat_history: list[tuple[str, str]] = []

    while True:
        q = input("Pergunta (ou 'sair'): ").strip()
        if q.lower() == "sair":
            break

        result = qa({"question": q, "chat_history": chat_history})
        answer = result["answer"]
        print("\n🤖", answer, "\n")


        chat_history.append((q, answer))
