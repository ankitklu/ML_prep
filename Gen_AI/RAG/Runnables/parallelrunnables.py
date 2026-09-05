from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

#n
from langchain_core.runnables import RunnableSequence, RunnableMap, RunnableParallel, RunnableLambda

load_dotenv()

prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question: {question}")

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

parser = StrOutputParser()


#Two different prompts

short_prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question in one line: {topic}")

detailed_prompt = ChatPromptTemplate.from_template("You are a helpful assistant. Answer the following question in 3-4 lines: {topic}")

topic = "Machine Learning"


chain = prompt | llm | parser

# formatted_short = short_prompt.format_message(topic = topic)
# response = llm.invoke(formatted_short)
# str_out = parser.parse(response.content)

# chain = RunnableParallel({
#     "short": short_prompt | llm | parser,
#     "detailed": detailed_prompt | llm | parser
# })

chain = RunnableParallel({
    "short": RunnableLambda(lambda x: x['short']) | short_prompt | llm | parser,
    "detailed": RunnableLambda(lambda x: x['detailed']) | detailed_prompt | llm | parser
})

## Runnable Passthroughs


# chain.invoke({"topic": "Machine Learning"})

#single topic input to the parallel chain
# result =chain.invoke({"topic": "Machine Learning"})


#multi topic input to the parallel chain
result =chain.invoke({
    "short": {"topic": "Machine Learning"}, 
    "detailed": {"topic": "Deep Learning"}
})


print(result["short"])
print("---------------------------------------------------------------------------------------------------------------------------------------------------")
print(result["detailed"])


 