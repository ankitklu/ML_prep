from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = ["What is the capital of India?", "Who is the richest person alive?"]

result = embeddings.embed_documents(texts)
print(str(result))