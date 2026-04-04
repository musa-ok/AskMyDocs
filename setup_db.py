import os
import sys
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

DATA_DIR = "./data/raw_pdfs"

if not os.path.exists(DATA_DIR):
    print(f"Error: Directory '{DATA_DIR}' not found.")
    sys.exit(1)

documents = []
for file in os.listdir(DATA_DIR):
    if file.endswith(".pdf"):
        print(f"Processing: {file}")
        loader = PyPDFLoader(os.path.join(DATA_DIR, file))
        documents.extend(loader.load())

if not documents:
    print("Error: No PDF files found.")
    sys.exit(1)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

vectorstore = QdrantVectorStore.from_documents(
    documents=splits,
    embedding=embeddings,
    path="./qdrant_db",
    collection_name="rag-chroma",
    force_recreate=True
)

print("Database setup completed.")