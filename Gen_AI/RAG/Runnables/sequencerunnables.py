from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = StrOutputParser()

chain = prompt | llm | parser

result =chain.invoke("Machine Learning")

print(result)