from langchain_community.document_loaders import WebBaseLoader

url = "https://en.wikipedia.org/wiki/Artificial_intelligence"

loader = WebBaseLoader(url)

data = loader.load()

print(data[0].page_content)