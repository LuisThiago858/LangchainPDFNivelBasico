from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv

load_dotenv()


loader = PyPDFLoader("Relatorio_Completo_Almater.pdf")
docs = RecursiveCharacterTextSplitter(
    chunk_size=1500, chunk_overlap=300
).split_documents(loader.load())

vectorstore = Chroma.from_documents(docs, embedding=OpenAIEmbeddings())


custom_prompt = PromptTemplate(
    input_variables=["context", "question"], 
    template="""
    Use os trechos abaixo do documento para responder com exatidão.
    Se a resposta estiver em forma de tabela, extraia o valor diretamente.

    Pergunta: {question}
    ========
    {context}
    ========
    Resposta:"""
)


qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model="gpt-4-turbo"),
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    chain_type_kwargs={"prompt": custom_prompt}
)


while True:
    pergunta = input("Pergunte algo sobre o seus indicadores a Gisa (ou digite 'sair'): ")
    if pergunta.lower() == "sair":
        break


    resposta = qa.invoke({"query": pergunta})

    print("\n🧠 Resposta:", resposta["result"])
    print("\n📄 Fontes utilizadas:")
    for i, doc in enumerate(resposta["source_documents"], 1):
        print(f"\nFonte {i}:\n{doc.page_content[:500]}...\n")

