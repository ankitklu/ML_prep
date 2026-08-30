from dotenv import load_dotenv
load_dotenv()

from langchain_google_genai import GoogleGenerativeAIEmbeddings

embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", dimensions=64)

result = embeddings.embed_query("What is the capital of India?")
print(str(result))
