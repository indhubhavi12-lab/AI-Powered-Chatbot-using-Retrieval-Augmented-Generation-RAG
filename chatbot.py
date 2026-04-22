from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import streamlit as st # Import streamlit for caching
from langchain_core.prompts import ChatPromptTemplate
import os

@st.cache_resource
def load_embeddings_model():
    """Loads the HuggingFace embeddings model once."""
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

@st.cache_resource
def load_faiss_index(embeddings_model):
    """Loads the FAISS index once."""
    return FAISS.load_local("embeddings/", embeddings_model, allow_dangerous_deserialization=True)

def get_response(query, chat_history=[], model_name="mixtral-8x7b-32768"):
    try:
        embeddings = load_embeddings_model()
        # Use absolute path detection for cloud environments
        base_path = os.path.dirname(__file__)
        index_path = os.path.join(base_path, "embeddings")
        db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

        # Retrieve context
        docs = db.similarity_search(query)
        context = "\n".join([doc.page_content for doc in docs])

        model = ChatGroq(model_name=model_name)

        # Improved prompt that considers chat history and context
        template = """You are an intelligent assistant for a company. 
Use the following pieces of context and the conversation history to answer the question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

History:
{history}

Question: {query}

Answer:"""
        
        prompt_template = ChatPromptTemplate.from_template(template)
        chain = prompt_template | model
        
        # Format history for the prompt
        formatted_history = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history[-3:]])
        response = chain.invoke({"context": context, "query": query, "history": formatted_history})
        return response.content
    except Exception as e:
        return f"⚠️ RAG Error: {str(e)}"