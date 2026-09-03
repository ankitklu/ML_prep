from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embedding_model = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embedding_model,
    persist_directory="chroma-db"
)

result = vectorstore.similarity_search("What is Python used for?", k=2)
for doc in result:
    print(f"Content: {doc.page_content}, Source: {doc.metadata['source']}")


retriever = vectorstore.as_retriever()

docs = retriever.invoke("Explain deep learning?")

print("Retrieved Documents:")
for doc in docs:
    print(f"Content: {doc.page_content}, Source: {doc.metadata['source']}")


