"""Simulated user: answers clarifying questions using only the hidden 'remaining shards'."""
from typing import List

from src.llm_client import call_llm

SIMULATOR_SYSTEM_PROMPT = """You are role-playing a USER who sent an under-specified request to an AI assistant.
The assistant is now asking you clarifying questions. You know your full intent, written below as FACTS.

Rules:
1. Answer using ONLY the facts. Never invent details that are not in them.
2. Answer only what the assistant asked. Do not volunteer facts it did not ask about.
3. If the facts do not cover a question, say you have no particular preference or that it is not specified.
4. Reply in one to three short, natural sentences, like a real user would.
5. Never solve the task yourself."""


def simulate_user_reply(original_request: str, questions: List[str], facts: List[str]) -> str:
    fact_block = "\n".join(f"- {fact}" for fact in facts)
    asked = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, 1))
    user_content = (
        f"Your original request: {original_request}\n\n"
        f"FACTS you know:\n{fact_block}\n\n"
        f"The assistant asks:\n{asked}\n\n"
        "Your reply:"
    )
    return call_llm(
        [
            {"role": "system", "content": SIMULATOR_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.0,
    )
