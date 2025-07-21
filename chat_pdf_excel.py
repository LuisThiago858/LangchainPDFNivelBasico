from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

import pandas as pd
from pathlib import Path
import re, os, shutil

load_dotenv()

# ---------- 1. Config --------- #
BASES = {
    "4": "indicadores_financeiros_almater.xlsx",
    # coloque outros arquivos se quiser
}

MONTH_ALIASES = {
    "Jan": "Janeiro", "Fev": "Fevereiro", "Mar": "Março", "Abr": "Abril",
    "Mai": "Maio", "Jun": "Junho", "Jul": "Julho", "Ago": "Agosto",
    "Set": "Setembro", "Out": "Outubro", "Nov": "Novembro", "Dez": "Dezembro",
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

# ---------- 2. Loader genérico p/ Excel ---------- #
def excel_to_documents(path: str) -> list[Document]:
    dfs = pd.read_excel(path, sheet_name=None, header=None)  # todas as abas, sem header fixo
    docs: list[Document] = []

    for sheet_name, raw in dfs.items():
        # 2.1 achar primeira linha não vazia
        first_row_idx = raw.first_valid_index()
        if first_row_idx is None:
            continue  # aba vazia
        context = str(raw.iloc[first_row_idx, 0]).strip()

        # 2.2 achar header na linha seguinte
        header_idx = first_row_idx + 1
        header = raw.iloc[header_idx].tolist()
        header = [str(h).strip() for h in header]

        # 2.3 dados => tudo após header
        data = raw.iloc[header_idx + 1 :].copy()
        data.columns = header
        data = data.dropna(how="all")  # remove linhas 100% vazias

        # 2.4 se a 1ª coluna chama "Mês", derretemos anos
        if "Mês" in data.columns:
            id_vars = ["Mês"]
            value_vars = [c for c in data.columns if re.fullmatch(r"\d{4}", str(c))]
            long_df = pd.melt(
                data,
                id_vars=id_vars,
                value_vars=value_vars,
                var_name="Ano",
                value_name="Saldo",
            ).dropna(subset=["Saldo"])

            for _, row in long_df.iterrows():
                mes_abrev = str(row["Mês"]).strip()
                mes_full = MONTH_ALIASES.get(mes_abrev, mes_abrev)
                ano = int(row["Ano"])
                saldo = row["Saldo"]

                text = (
                    f"Contexto: {context}\n"
                    f"Mês: {mes_full} ({mes_abrev})\n"
                    f"Ano: {ano}\n"
                    f"Saldo: {saldo}"
                )
                meta = {"sheet": sheet_name, "mes": mes_abrev, "ano": ano}
                docs.append(Document(page_content=text, metadata=meta))
        else:
            # fallback: guarda a tabela inteira como texto
            texto_tabela = data.to_csv(index=False, sep="|")
            docs.append(Document(page_content=f"{context}\n{texto_tabela}",
                                 metadata={"sheet": sheet_name}))
    return docs

# ---------- 3. Pipeline RAG ---------- #
def build_qa(file_path: str) -> RetrievalQA:
    store_dir = f"chromax_{Path(file_path).stem}"
    if os.path.exists(store_dir):
        vector = Chroma(persist_directory=store_dir,
                        embedding_function=OpenAIEmbeddings())
    else:
        docs = excel_to_documents(file_path)

        vector = Chroma.from_documents(
            docs,
            embedding=OpenAIEmbeddings(),
            persist_directory=store_dir,
        )

    retriever = vector.as_retriever(search_kwargs={"k": 8})
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4o-mini"),
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT},
    )

# ---------- 4. CLI simples ---------- #
file_path = BASES["4"]
# se quiser reconstruir o índice quando a planilha mudar:
# shutil.rmtree(f"chromax_{Path(file_path).stem}", ignore_errors=True)

qa = build_qa(file_path)
print("Base carregada!\n")

while True:
    q = input("Pergunta (ou 'sair'): ").strip()
    if q.lower() == "sair":
        break
    out = qa.invoke({"query": q})
    print("🧠", out["result"], "\n")

    print("\n📄 Fonte:")
    for d in out["source_documents"]:
        print(
            f"- Aba: {d.metadata.get('sheet')} | "
            f"Mês: {d.metadata.get('mes')} | "
            f"Ano: {d.metadata.get('ano')} | "
            f"Trecho: {d.page_content.splitlines()[0][:80]}…"
        )

