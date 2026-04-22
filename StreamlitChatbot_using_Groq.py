import streamlit as st
from groq import Groq
from chatbot import get_response as get_rag_response

import os
from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SYSTEM_PROMPT = "You are a helpful assistant."

if not GROQ_API_KEY:
    st.error("GROQ_API_KEY not found. Please check your .env file.")
    st.stop()

client = Groq(api_key=GROQ_API_KEY)

st.set_page_config(page_title="Groq AI Assistant", page_icon="💬")
st.title("💬 AI Chatbot (Groq)")

# --- Sidebar ---
with st.sidebar:
    st.header("Settings")
    model_option = st.selectbox(
        "Choose a model:",
        ["llama-3.1-8b-instant", "mixtral-8x7b-32768", "llama3-70b-8192"],
        index=0
    )
    use_rag = st.checkbox("Use Company Knowledge Base (RAG)", value=False)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Initialize Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User Input ---
if prompt := st.chat_input("Ask something..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."): # Show a loading spinner
        try:
            if use_rag:
                # Use the RAG logic from chatbot.py
                # Pass existing message history (excluding the current prompt)
                reply = get_rag_response(prompt, chat_history=st.session_state.messages[:-1], model_name=model_option)
            else:
                # Standard Groq API call
                api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
                    {"role": m["role"], "content": m["content"]} for m in st.session_state.messages
                ]
                response = client.chat.completions.create(
                    model=model_option,
                    messages=api_messages
                )
                reply = response.choices[0].message.content

        except Exception as e:
            reply = "⚠️ Error: Could not get a response. Please check your API key, model name, or network connection."
            st.error(f"An error occurred: {e}") # Display error in Streamlit
            print(f"Groq API error: {e}") # Log error to console

    # --- Display Assistant Response ---
    if reply: # Only append and display if a reply was generated
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
