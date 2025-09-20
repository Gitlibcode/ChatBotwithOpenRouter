import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
import os

# Safely retrieve OpenRouter API key
OPENROUTER_API_KEY = st.secrets.get("api_key") or os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error("❌ OpenRouter API key not found. Please set it in Streamlit secrets or environment variables.")
    st.stop()

# Custom wrapper for OpenRouter
class ChatOpenRouter(ChatOpenAI):
    def __init__(self, openai_api_key=None, **kwargs):
        super().__init__(
            base_url="https://openrouter.ai/api/v1",
            openai_api_key=openai_api_key,
            **kwargs
        )

# Initialize session state
if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]

# Initialize OpenRouter model
try:
    llm = ChatOpenRouter(
        openai_api_key=OPENROUTER_API_KEY,
        model_name="anthropic/claude-3.7-sonnet:thinking",
        max_tokens=512
    )
except Exception as e:
    st.error(f"❌ Failed to initialize LLM: {e}")
    st.stop()

# Create conversation chain
conversation = ConversationChain(memory=st.session_state.buffer_memory, llm=llm)

# UI setup
st.title("🗣️ Conversational Chatbot")
st.subheader("㈻ Simple Chat Interface for LLMs")

# Handle user input
if prompt := st.chat_input("Your question"):
    st.session_state.messages.append({"role": "user", "content": prompt})

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Generate assistant response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = conversation.predict(input=prompt)
                st.write(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"❌ Error generating response: {e}")
