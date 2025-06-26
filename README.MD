# 📚 Chat-PDF-Retriever

Um chat em linha de comando que responde perguntas sobre **vários relatórios em PDF** usando LangChain, um banco vetorial Chroma e o modelo GPT-4-turbo da OpenAI.

> “Escolha a base → faça perguntas → receba resposta **+** trechos do PDF usados como fonte.”

---

## ✨ Funcionalidades

| Recurso | Descrição |
|---------|-----------|
| **Seleção dinâmica de PDF** | O usuário escolhe 1, 2 ou 3 e o script carrega o PDF correspondente. |
| **Indexação com cache** | Embeddings ficam salvos em `chroma_<nome-do-pdf>/`; o mesmo PDF abre instantaneamente na próxima execução. |
| **Prompt customizado** | Instrui o GPT-4 a extrair valores exatos de tabelas quando necessário. |
| **Fontes exibidas** | Mostra até 500 caracteres dos trechos que fundamentaram a resposta — transparência total. |
| **Modo conversa** | Loop de perguntas ilimitadas até o usuário digitar `sair`. |

---

## 🛠️ Dependências

- Python ≥ 3.9  
- `langchain`, `langchain-community`, `langchain-openai`  
- `chromadb`, `pypdf`, `tiktoken`  
- `python-dotenv` (para gerir `OPENAI_API_KEY`)

Instale tudo com:

```bash
pip install -r requirements.txt

```bash
python chat_pdf2.py

