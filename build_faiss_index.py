import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def build_index(data_path="data/data.txt.txt", index_path="embeddings/"):
    """
    Builds a FAISS index from the provided text data using HuggingFaceEmbeddings.
    """
    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.from_texts(texts, embeddings)

    if not os.path.exists(index_path):
        os.makedirs(index_path)
    db.save_local(index_path)
    print(f"FAISS index built and saved to {index_path}")

if __name__ == "__main__":
    build_index()
