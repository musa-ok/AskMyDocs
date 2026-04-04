"""
Ingest: only files under BASE_DIR/data (whatever the user placed there via Streamlit or otherwise).
Each run deletes the Qdrant collection and rebuilds from the current folder contents — no baked-in filenames.
"""
import os

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
DATA_DIR = os.path.join(BASE_DIR, "data")
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")
COLLECTION_NAME = "rag-chroma"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def _delete_collection_if_exists(client: QdrantClient) -> None:
    try:
        names = {c.name for c in client.get_collections().collections}
    except Exception:
        names = set()
    if COLLECTION_NAME in names:
        client.delete_collection(collection_name=COLLECTION_NAME)


def _load_documents_from_data_dir(data_dir: str) -> list:
    """Recursively load all PDF and TXT files under data_dir (no hard-coded file names)."""
    if not os.path.isdir(data_dir):
        os.makedirs(data_dir, exist_ok=True)
        return []

    documents = []

    pdf_loader = DirectoryLoader(
        data_dir,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        recursive=True,
        show_progress=False,
    )
    txt_loader = DirectoryLoader(
        data_dir,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
        recursive=True,
        show_progress=False,
    )

    documents.extend(pdf_loader.load())
    documents.extend(txt_loader.load())

    return documents


def ingest_docs():
    """
    1) Delete rag-chroma in Qdrant (full reset).
    2) Scan only BASE_DIR/data for .pdf / .txt.
    3) Split with chunk_size=1000, overlap=200, embed with all-MiniLM-L6-v2, persist under qdrant_db.
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(QDRANT_PATH, exist_ok=True)

    client = QdrantClient(path=QDRANT_PATH)
    _delete_collection_if_exists(client)

    documents = _load_documents_from_data_dir(DATA_DIR)

    if not documents:
        return None

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    splits = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    vectorstore = QdrantVectorStore.from_documents(
        documents=splits,
        embedding=embeddings,
        path=QDRANT_PATH,
        collection_name=COLLECTION_NAME,
    )
    return vectorstore


if __name__ == "__main__":
    ingest_docs()
