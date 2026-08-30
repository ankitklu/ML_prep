from dotenv import load_dotenv
load_dotenv()

# this is the class that lets us talk to Gemini through LangChain's unified interface
from langchain.chat_models import init_chat_model

# these three message types are how LangChain represents a conversation:
# SystemMessage sets the assistant's behavior, HumanMessage is what we say,
# AIMessage is what the model says back
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# spin up the Gemini model we tested earlier (2.6-flash is the current live one)
model = init_chat_model("google_genai:gemini-3.6-flash")

# history is just a plain list of message objects - we keep appending to it
# so every new call includes the full conversation so far, which is how the
# model "remembers" what was said earlier
history = [
    SystemMessage(content="You are a helpful assistant. Keep answers concise."),
]

print("Chatbot is ready! Type 'exit' to quit.\n")

# loop forever, one turn per iteration, until the user wants to stop
while True:
    # grab whatever the user types in the terminal
    user_input = input("You: ")

    # let the user bail out whenever they want
    if user_input.strip().lower() in ("exit", "quit"):
        print("Bot: Goodbye!")
        break

    # record the user's message in history before we send anything to the model
    history.append(HumanMessage(content=user_input))

    # send the *entire* history (system + all past turns) so the model has context
    response = model.invoke(history)

    # save the model's reply as an AIMessage so it's part of history for next time
    history.append(AIMessage(content=response.content))

    # show the reply to the user
    print("Bot:", response.content)
