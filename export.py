# app.py

import json
import logging
import re
from typing import List

from pydantic import BaseModel, RootModel, ValidationError

# ← use the chat interface
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage

from langgraph.graph import StateGraph, START, END

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SubQuery(BaseModel):
    text: str


class AgentState(BaseModel):
    user_query: str | None = None
    sub_queries: List[SubQuery] = []
    ranked_queries: List[SubQuery] = []
    previous_qas: List[str] = []


class SubqueriesModel(RootModel[List[str]]):
    """Validates that the LLM output is a JSON array of strings."""


class RankedModel(RootModel[List[str]]):
    """Validates that the LLM output is a JSON array of strings."""


def get_user_query(state: AgentState) -> dict:
    logger.info(f"State before user input: {state}")
    q = input("Enter your query: ")
    return {"user_query": q}


def classify_query(state: AgentState) -> dict:
    logger.info(f"State before classification: {state}")

    system = SystemMessage(
        content=(
            "Split the following user question into its sub‑queries, "
            "and respond *only* with a JSON array of strings. "
            "If there are no sub‑queries, return an empty array."
        )
    )
    user = HumanMessage(content=state.user_query or "")
    resp = llm.invoke([system, user])
    raw = resp.content

    try:
        model = SubqueriesModel.model_validate_json(raw)
        strings = model.root
    except ValidationError:
        try:
            data = json.loads(raw)
            model = SubqueriesModel.model_validate(data)
            strings = model.root
        except Exception:
            strings = [
                line.strip("- ").strip()
                for line in raw.splitlines()
                if line.strip()
            ]

    sub_qs = [SubQuery(text=s) for s in strings]
    return {"sub_queries": sub_qs}


def rank_queries(state: AgentState) -> dict:
    logger.info(f"State before ranking: {state}")
    texts = [sq.text for sq in state.sub_queries]

    system = SystemMessage(
        content=(
            "Given these sub‑queries, respond *only* with a JSON array of strings. "
            "Rank them in the most logical order to answer. "
            "Do *not* include any answers or extra commentary."
        )
    )
    user = HumanMessage(content=json.dumps(texts, ensure_ascii=False))
    resp = llm.invoke([system, user])
    raw = resp.content

    try:
        model = RankedModel.model_validate_json(raw)
        strings = model.root
    except ValidationError:
        match = re.search(r'(\[.*?\])', raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                model = RankedModel.model_validate(data)
                strings = model.root
            except Exception:
                strings = []
        else:
            strings = []

    ranked = [SubQuery(text=s) for s in strings]
    return {"ranked_queries": ranked}


def setup_llm() -> ChatOllama:
    return ChatOllama(
        model="ollama3.2-3B",
        base_url="http://host.docker.internal:11434"
    )


def main():
    global llm
    llm = setup_llm()

    workflow = StateGraph(AgentState)
    workflow.add_node("input",    get_user_query)
    workflow.add_node("classify", classify_query)
    workflow.add_node("rank",     rank_queries)

    workflow.add_edge(START,      "input")
    workflow.add_edge("input",    "classify")
    workflow.add_edge("classify", "rank")
    workflow.add_edge("rank",     END)

    agent = workflow.compile()
    result = agent.invoke({})

    final_state = AgentState(**result)
    print("\nFinal Agent State:")
    print(final_state.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
