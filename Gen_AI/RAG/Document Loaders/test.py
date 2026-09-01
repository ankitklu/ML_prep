from langchain_community.document_loaders import TextLoader

data = TextLoader("notes.txt")

docs = data.load()

print(docs)