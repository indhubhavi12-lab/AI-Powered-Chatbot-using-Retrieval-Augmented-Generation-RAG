from langchain.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

def get_response(query, model_name="mixtral-8x7b-32768"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    # Load the vector database with safety flag for local loading
    db = FAISS.load_local("embeddings/", embeddings, allow_dangerous_deserialization=True)

    docs = db.similarity_search(query)
    context = "\n".join([doc.page_content for doc in docs])

    model = ChatGroq(model_name=model_name)

    # Integrated prompt engineering for better response quality
    prompt = f"""You are an intelligent assistant.
Answer clearly and concisely based on the context provided.

Context:
    {context}

    Question: {query}

Answer:"""
    response = model.predict(prompt)
    return response