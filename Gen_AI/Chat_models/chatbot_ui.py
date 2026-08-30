import streamlit as st

# load the .env file so GOOGLE_API_KEY is available
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

st.title("Gemini Chatbot")

# streamlit reruns the whole script on every interaction, so we stash the
# conversation in session_state - that's the only way it survives a rerun
if "history" not in st.session_state:
    st.session_state.history = [
        SystemMessage(content="You are a helpful assistant. Keep answers concise."),
    ]

# cache the model so we don't reconnect to Gemini on every single rerun
if "model" not in st.session_state:
    st.session_state.model = init_chat_model("google_genai:gemini-3.6-flash")

# replay everything said so far, skipping the system message since that's
# just an instruction for the model, not something the user should see
for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# the chat box at the bottom of the page - returns None until the user submits something
user_input = st.chat_input("Type your message...")

if user_input:
    # show the user's own message immediately
    with st.chat_message("user"):
        st.markdown(user_input)

    # add it to history before calling the model, so the model sees it too
    st.session_state.history.append(HumanMessage(content=user_input))

    # show a spinner in the assistant bubble while Gemini is generating the reply
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = st.session_state.model.invoke(st.session_state.history)
        st.markdown(response.content)

    # save the reply to history for the next turn
    st.session_state.history.append(AIMessage(content=response.content))
