from dotenv import load_dotenv
load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1"
    # task="text-generation",
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is the capital of India ?")
print(response.content)