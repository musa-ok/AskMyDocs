"""
LangGraph workflow: router entry, RAG path, optional web search, chat-only branch, and generation grading.
"""
import sys
import os
from dotenv import load_dotenv

project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.insert(0, project_dir)

from langgraph.graph import END, StateGraph
from app.core.nodes import (
    GraphState,
    retrieve,
    grade_documents,
    generate,
    web_search,
    rewrite_question,
    conversational_reply,
)
from app.core.router import question_router
from app.core.grader import hallucination_grader, answer_grader

load_dotenv()


def route_question(state):
    """LLM router: chat, vectorstore retrieve, or web search."""
    question = state["question"]
    source = question_router.invoke({"question": question})

    if source.datasource == "chat":
        return "conversational_reply"
    if source.datasource == "websearch":
        return "web_search"
    if source.datasource == "vectorstore":
        return "retrieve"
    return "conversational_reply"


def decide_to_generate(state):
    """After grading docs: optionally rewrite query for web search before generate."""
    web_search_flag = state.get("web_search", False)
    search_count = state.get("search_count", 0)

    if web_search_flag:
        if search_count < 2:
            return "rewrite_question"
        else:
            return "generate"
    else:
        return "generate"


def grade_generation_v_documents_and_question(state):
    """Hallucination and answer grading loop labels for conditional edges."""
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    search_count = state.get("search_count", 0)

    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    grade = score.binary_score

    if grade == "yes":
        score = answer_grader.invoke({"question": question, "generation": generation})
        answer_grade = score.binary_score

        if answer_grade == "yes":
            return "useful"
        else:
            if search_count < 2:
                return "not useful"
            else:
                return "useful"
    else:
        return "not supported"


workflow = StateGraph(GraphState)

workflow.add_node("conversational_reply", conversational_reply)
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("web_search", web_search)
workflow.add_node("generate", generate)

workflow.set_conditional_entry_point(
    route_question,
    {
        "conversational_reply": "conversational_reply",
        "web_search": "web_search",
        "retrieve": "retrieve",
    },
)

workflow.add_edge("conversational_reply", END)

workflow.add_edge("retrieve", "grade_documents")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        "rewrite_question": "rewrite_question",
        "generate": "generate",
    },
)

workflow.add_edge("rewrite_question", "web_search")

workflow.add_edge("web_search", "grade_documents")

workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "not useful": "rewrite_question",
        "useful": END,
    },
)

app = workflow.compile()