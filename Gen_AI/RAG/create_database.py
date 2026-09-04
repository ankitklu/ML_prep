#load pdf
# divide into chunks 
# create embeddings for each chunk
# store embeddings in vector database
#retrieve relevant chunks based on user query

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()

data = PyPDFLoader("Document Loaders/deeplearning.pdf")
docs = data.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# runs locally (sentence-transformers model downloaded once, cached after) -
# no API calls, so no rate limits when embedding a whole book
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory="chroma_db"
)

print("Vector database created and embeddings stored successfully.")