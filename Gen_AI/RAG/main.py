from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

model = ChatMistralAI(model="mistral-small-2506")

template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that answers questions based on the provided documents."),
    ("human", "{question}")
])

data = PyPDFLoader("Document Loaders/GRU.pdf")

docs = data.load()

prompt = template.format_messages(question=docs[0].page_content)

response = model.invoke(prompt)
print(response.content)
