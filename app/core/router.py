"""LLM-based router: vectorstore vs websearch vs chat-only path."""
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0
)

class RouterQuery(BaseModel):
    datasource: Literal["vectorstore", "websearch", "chat"] = Field(
        ...,
        description=(
            "vectorstore: default for any factual/doc/technical question (user uploads); "
            "websearch: only when live web data is required; "
            "chat: ONLY pure greeting/small talk with zero information request"
        ),
    )

structured_llm_router = llm.with_structured_output(RouterQuery)

system_prompt = """
You are a strict message router. Output exactly one of: vectorstore | websearch | chat.

## Non-negotiable rules

1) **Ignore conversational fillers.** Words or prefixes such as "Peki", "Acaba", "Kanka", "Ya",
   "Hey", "Bi dakika", "Şey", "Yani", "Sence", English fillers like "So", "Well", "BTW", or similar
   MUST NOT push the message toward **chat**. Classify based on whether there is a real information
   request, not on tone or slang.

2) **vectorstore** — Use when the user asks for ANY substantive information that could be answered
   from their uploaded documents or knowledge base: facts, laws, policies, company rules, procedures,
   definitions, technical details, data, "what does the doc say", summaries of content, comparisons,
   how-to grounded in documentation, or any question that expects a factual or document-based answer.
   If the message contains BOTH fillers AND such a request, you MUST choose **vectorstore**.

3) **websearch** — Use ONLY when the answer clearly requires **current** public web information
   (breaking news, today's weather, live prices, sports results, "what happened today", etc.)
   that is unlikely to live in static uploads. Do not use **websearch** just because the user sounds casual.

4) **chat** — Use ONLY for messages that are **purely** social with **no** information request:
   standalone greetings ("hi", "merhaba"), thanks with no follow-up question, "how are you" with no
   other task, idle banter with no ask. If there is even one factual, procedural, or document-related
   question hidden in the message, you MUST NOT choose **chat**.

5) **Default when unsure** — If you are uncertain between **chat** and anything else, choose
   **vectorstore**. If you are uncertain between **vectorstore** and **websearch**, choose
   **vectorstore** unless the need for live web data is obvious.

## Quick check before you answer
- Is the user asking for information, a rule, a fact, or content from files? → **vectorstore** (or **websearch** only if live web is clearly required).
- Is it ONLY hello/thanks/small talk with no ask? → **chat**.
"""

route_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{question}")
     ]
)

question_router = route_prompt | structured_llm_router

