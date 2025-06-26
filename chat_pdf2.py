from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter #quebra o texto em pedaços menores.
from langchain_community.vectorstores import Chroma #Cria um banco de dados de vetores
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

BASES = {
    "1": "Relatorio_Completo_Almater.pdf",
    "2": "Relatorio_Completo_SCVP.pdf",
    "3": "Proximo_Cliente.pdf",
}


prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use os trechos abaixo do documento para responder com exatidão.
    Se a resposta estiver em forma de tabela, extraia o valor diretamente.
    Se não tiver certeza sobre um dado numérico, indicador, métrica ou
    valor referente a um período específico, não faça suposições nem
    forneça estimativas sem fonte. Em vez disso, responda de forma direta
    que os dados necessários não estão disponíveis ou não podem ser confirmados neste momento.

    Pergunta: {question}
    ========
    {context}
    ========
    Resposta:"""
)


def build_qa(pdf_path: str) -> RetrievalQA:

    store_dir = f"chroma_{Path(pdf_path).stem}"
    if Path(store_dir).exists():
        vector = Chroma(persist_directory=store_dir, embedding_function=OpenAIEmbeddings())
    else:
        # Carrega e indexa
        docs = RecursiveCharacterTextSplitter(
            chunk_size=1500, chunk_overlap=300
        ).split_documents(PyPDFLoader(pdf_path).load())
        vector = Chroma.from_documents(
            docs,
            embedding=OpenAIEmbeddings(),
            persist_directory=store_dir,
        )
        vector.persist()

    # 2. Cria a cadeia QA
    return RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model="gpt-4-turbo"),
        retriever=vector.as_retriever(),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


escolha = None
while escolha not in BASES:
    escolha = input("Escolha a base com base no ID (1, 2 ou 3) ou 'sair': ").strip()
    if escolha == "sair":
        exit()

qa = build_qa(BASES[escolha])
print(f" Base {escolha} carregada: {BASES[escolha]}")

# ---------- Loop de perguntas ----------
while True:
    pergunta = input("\nPergunte algo a mais para a Gisa (ou escreva 'sair' para parar): ").strip()
    if pergunta.lower() == "sair":
        break

    resposta = qa.invoke({"query": pergunta})
    print("\n🧠 Resposta:", resposta["result"])

    print("\n📄 Fontes:")
    for i, doc in enumerate(resposta["source_documents"], 1):
        print(f"\nFonte {i}:\n{doc.page_content[:500]}...\n")
