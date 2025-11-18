import streamlit as st
import os
import time
import base64
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.chat_history import InMemoryChatMessageHistory
from dotenv import load_dotenv

# Set up environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Custom CSS (Kept your styling)
custom_css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    :root {
        --primary-color: #0d47a1;
        --secondary-color: #ff6f00;
        --accent-color: #f5f5f5;
        --text-color: #333333;
        --chat-bg: #ffffff;
        --header-bg: var(--primary-color);
        --header-text-color: #ffffff;
    }
    .main { color: var(--text-color); }
    .stApp { color: var(--text-color); background-color: var(--accent-color); }
    .header-container {
        background: var(--header-bg); color: var(--header-text-color);
        padding: 2rem; border-radius: 0 0 20px 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2); margin-bottom: 2rem;
        display: flex; align-items: center; justify-content: space-between;
    }
    .header-text h1, .header-text p { color: var(--header-text-color) !important; }
    .header-image img { max-width: 150px; border-radius: 10px; box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2); }
    .title-font { font-size: 60px; font-family: 'Roboto', sans-serif !important; font-weight: 700; }
    .chat-container { max-width: 800px; margin: 0 auto; padding: 20px; }
    .stChatMessage { background-color: var(--chat-bg); border: 1px solid #ddd; border-radius: 15px; padding: 15px; margin: 10px 0; color: var(--text-color); }
    .stChatMessage p { color: var(--text-color) !important; }
    .stTextInput input { border-radius: 25px; border: 2px solid var(--primary-color); padding: 10px 20px; background-color: #ffffff; }
    div.stButton > button:first-child { background-color: var(--secondary-color); color: #ffffff; border-radius: 25px; padding: 0.5rem 2rem; border: none; }
    div.stButton > button:hover { background-color: var(--primary-color); transform: translateY(-2px); }
    .legal-disclaimer { background-color: rgba(255, 243, 205, 0.5); border-left: 4px solid var(--secondary-color); padding: 1rem; margin: 1rem 0; border-radius: 4px; color: #555; }
    .warning-message { background-color: #fff3cd; color: #856404; padding: 10px; border-radius: 4px; margin: 5px 0; border: 1px solid #ffeeba; }
</style>
"""

st.set_page_config(page_title="Nyay Mitra", layout="wide", initial_sidebar_state="collapsed")
st.markdown(custom_css, unsafe_allow_html=True)

# Header
try:
    image_path = "saul.jpg"
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
        st.markdown(f"""
            <div class="header-container">
                <div class="header-text">
                    <h1 style="color: white; font-size: 2.5rem;"><span class="title-font">Nyay Mitra</span></h1>
                    <p style="color: #e0e0e0; font-size: 1.2rem;">Your AI assistant for information on Indian Law.</p>
                </div>
                <div class="header-image"><img src="data:image/jpeg;base64,{encoded_image}" alt="Legal Bot"></div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""<div class="header-container"><div class="header-text"><h1>Nyay Mitra</h1></div></div>""", unsafe_allow_html=True)
except Exception:
    st.markdown("""<div class="header-container"><div class="header-text"><h1>Nyay Mitra</h1></div></div>""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="legal-disclaimer">
    <h4>⚠️ Legal Information Disclaimer</h4>
    <ul>
        <li><b>For Information Only:</b> This AI provides general information on Indian law, not legal advice.</li>
        <li><b>Not a Substitute:</b> It does not replace consultation with a qualified advocate.</li>
        <li><b>Check Applicability:</b> Information may not be up-to-date or apply to your specific situation.</li>
        <li><b>Do Not Rely:</b> This information cannot be the basis for legal decisions.</li>
        <li><b>Seek Counsel:</b> Always consult a qualified legal professional for specific legal matters.</li>
    </ul>
</div>
""", unsafe_allow_html=True)

# Initialize State
if "messages" not in st.session_state: st.session_state.messages = []
if "chat_history" not in st.session_state: st.session_state.chat_history = InMemoryChatMessageHistory()

def reset_conversation():
    st.session_state.messages = []
    st.session_state.chat_history.clear()

# Load Vector DB
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'}, encode_kwargs={'normalize_embeddings': True})
    db = FAISS.load_local("vector_db", embeddings, allow_dangerous_deserialization=True)
    # REDUCED k to 2 to prevent "Bad Request" errors from too much text
    db_retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 2})
except Exception as e:
    st.error(f"Error loading vector database: {e}")
    st.stop()

# CLEAN Prompt Template (Removed the Llama 2 special tags)
prompt_template = """
You are a legal assistant for Indian Law.

Strict Rules:
1. NEVER provide specific legal advice.
2. Only provide information based on Indian Law (Acts, Constitution, IPC).
3. If the answer is not in the context, state that you do not have that information.
4. Use clear, professional language.

CONTEXT: {context}

CHAT HISTORY: {chat_history}

QUESTION: {question}

ANSWER:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])

# Initialize LLM with the SAFE model
# If this fails, check your GROQ_API_KEY in secrets
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192")

def format_chat_history():
    history_text = ""
    for msg in st.session_state.chat_history.messages[-4:]:
        if isinstance(msg, HumanMessage): history_text += f"Human: {msg.content}\n"
        elif isinstance(msg, AIMessage): history_text += f"Assistant: {msg.content}\n"
    return history_text

# Chat Interface
st.markdown('<div class="chat-container">', unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message.get("role"), avatar="👤" if message.get("role") == "user" else "⚖️"):
        st.write(message.get("content"))

input_prompt = st.chat_input("Ask your question about Indian law...")

if input_prompt:
    with st.chat_message("user", avatar="👤"): st.write(input_prompt)
    st.session_state.messages.append({"role": "user", "content": input_prompt})
    st.session_state.chat_history.add_user_message(input_prompt)

    with st.chat_message("assistant", avatar="⚖️"):
        with st.status("Searching Indian legal documents...", expanded=True):
            try:
                docs = db_retriever.invoke(input_prompt)
                context = "\n\n".join([doc.page_content for doc in docs])
                formatted_prompt = prompt.format(context=context, question=input_prompt, chat_history=format_chat_history())
                
                response = llm.invoke(formatted_prompt)
                response_text = response.content
                
                st.write(response_text)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
                response_text = "I apologize, but I encountered an error while processing your request. Please try again."

    st.session_state.messages.append({"role": "assistant", "content": response_text})
    st.session_state.chat_history.add_ai_message(response_text)

    col1, col2, col3 = st.columns([4, 1, 4])
    with col2:
        st.button('🗑️ Clear Chat', on_click=reset_conversation, type="secondary", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)
