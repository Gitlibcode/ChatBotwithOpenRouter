# chatbot.py

import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
import os

# Set your OpenRouter API key
OPENROUTER_API_KEY = st.secrets['api_key']
#os.environ["OPENROUTER_API_KEY"] = "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"

# Custom wrapper for OpenRouter
class ChatOpenRouter(ChatOpenAI):
    def __init__(self, openai_api_key=None, **kwargs):
        openai_api_key = openai_api_key or os.environ.get("OPENROUTER_API_KEY")
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )

# Initialize session state
if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Initialize OpenRouter model
llm = ChatOpenRouter(model_name="anthropic/claude-3.7-sonnet:thinking",
max_tokens=512
)

# Create conversation chain
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

# UI setup
st.title("🗣️ Conversational Chatbot")
st.subheader("㈻ Simple Chat Interface for LLMs")

if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = conversation.predict(input=prompt)
            st.write(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
