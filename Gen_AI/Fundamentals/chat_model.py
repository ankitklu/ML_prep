from dotenv import load_dotenv
load_dotenv()


from langchain.chat_models import init_chat_model

model = init_chat_model("google_genai:gemini-3.6-flash")
# print(model)


response = model.invoke("Hello, how are you?")
print(response.content)
