from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

data = PyPDFLoader("GRU.pdf")

docs = data.load()

splitter = TokenTextSplitter(chunk_size=1000, chunk_overlap=10)

chunks = splitter.split_documents(docs)

for chunk in chunks:
    print(chunk.page_content)
    print("--------------------------------------------------")
