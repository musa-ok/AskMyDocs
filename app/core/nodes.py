import os
from typing import List, TypedDict
from rich.console import Console
from rich.panel import Panel
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from .grader import retrieval_grader
from .utils import trace_performance
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults

console = Console()
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")

client = QdrantClient(path=QDRANT_PATH)

if not client.collection_exists("rag-chroma"):
    console.print("[bold yellow]⚠️ Koleksiyon bulunamadı, sıfırdan oluşturuluyor...[/bold yellow]")
    client.create_collection(
        collection_name="rag-chroma",
        vectors_config=VectorParams(size=384, distance=Distance.COSINE),
    )

vectorstore = QdrantVectorStore(
    client=client,
    collection_name="rag-chroma",
    embedding=embeddings
)

class GraphState(TypedDict, total=False):
    question: str
    generation: str
    web_search: bool
    documents: List[Document]
    search_count: int
    rewritten_question: str
    chat_mode: bool
    is_conversation_opening: bool

@trace_performance
def retrieve(state):
    console.print(Panel("[bold cyan]🔍 Belge Veritabanında Arama Yapılıyor...[/bold cyan]", border_style="blue", title="[RETRIEVE]"))
    question = state["question"]
    retriever = vectorstore.as_retriever()
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}

@trace_performance
def grade_documents(state):
    console.print(Panel("[bold yellow]⚖️ Belgeler Uygunluk Testinden Geçiyor...[/bold yellow]", border_style="yellow", title="[GRADER]"))
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = False

    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if score.binary_score == "yes":
            filtered_docs.append(d)

    if not filtered_docs:
        console.print("[bold red]❌ Alakalı belge bulunamadı, Web Search tetikleniyor...[/bold red]")
        web_search = True
    else:
        console.print(f"[bold green]✅ {len(filtered_docs)} adet alakalı belge bulundu.[/bold green]")
        web_search = False

    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def decide_to_generate(state):
    web_search_flag = state.get("web_search", False)
    search_count = state.get("search_count", 0)

    if web_search_flag and search_count < 2:
        return "rewrite_question"
    else:
        return "generate"

@trace_performance
def generate(state):
    console.print(Panel("[bold green]✍️ Gemini 2.5 Flash Yanıt Üretiyor...[/bold green]", border_style="green", title="[GENERATOR]"))
    question = state["question"]
    documents = state["documents"]
    context = "\n\n".join([d.page_content for d in documents]) if documents else "No relevant documents found."

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3, max_retries=5)

    prompt = (
        "You are a helpful assistant. Answer the user's question strictly using the provided context. "
        "If the answer is not in the context, state that you do not have this information. "
        "Provide the final answer in the same language as the user's question.\n\n"
        f"CONTEXT:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    response = llm.invoke(prompt)
    return {"documents": documents, "question": question, "generation": response.content}

@trace_performance
def web_search(state):
    console.print(Panel("[bold red]🌐 Tavily ile İnternet Araması Başlatıldı...[/bold red]", border_style="red", title="[WEB SEARCH]"))
    question = state["question"]
    documents = state.get("documents", [])
    search_count = state.get("search_count", 0) + 1
    search_query = state.get("rewritten_question", question)
    web_search_tool = TavilySearchResults(k=3)
    docs = web_search_tool.invoke({"query": search_query})
    web_results = "\n".join([d["content"] for d in docs])
    web_document = Document(page_content=f"Web Results:\n{web_results}")
    return {"documents": documents + [web_document], "question": question, "search_count": search_count}

@trace_performance
def rewrite_question(state):
    console.print(Panel("[bold magenta]📝 Arama Sorgusu Optimize Ediliyor...[/bold magenta]", border_style="magenta", title="[REWRITER]"))
    question = state["question"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    system_prompt = "You are a search query optimizer. Convert the user question into a short, effective search engine query. Return ONLY the optimized query text."
    prompt = PromptTemplate(template=system_prompt + "\n\nQuestion: {question}\n\nSearch Query:", input_variables=["question"])
    chain = prompt | llm
    response = chain.invoke({"question": question})
    return {"rewritten_question": response.content, "question": question}

@trace_performance
def conversational_reply(state):
    console.print(Panel("[bold white]💬 Samimi Sohbet Yanıtı Hazırlanıyor...[/bold white]", border_style="white", title="[CONVERSATIONAL]"))
    question = state["question"]
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.8, max_retries=3)
    system = """You are the user's close friend and 'kanka'. 
- NEVER use cliché phrases like 'I am an AI' or 'I don't have feelings'.
- Be witty, warm, and natural. 
- Always reply in the SAME language as the user's question (Turkish).
- If asked 'How are you?', reply like a human friend (e.g., 'I'm doing great, literally flying on this M4 Air's neural engine!').

User's message:
{question}

Buddy's Response:"""
    prompt = PromptTemplate(template=system, input_variables=["question"])
    response = (prompt | llm).invoke({"question": question})
    return {
        "question": question,
        "generation": response.content,
        "documents": [],
        "web_search": False,
        "search_count": 0,
        "chat_mode": True,
    }