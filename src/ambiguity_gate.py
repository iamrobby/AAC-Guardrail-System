from src.llm_client import call_llm

AMBIGUITY_PROMPT = """You are an Ambiguity Detector for an AI assistant.

Your only job is to decide whether the user's request is clear enough to answer, or whether important details are missing.

Rules:
1. If the request is clear and specific enough to give a correct answer → reply with exactly: CLEAR
2. If important details are missing → ask 1 or 2 short, precise clarifying questions.
3. Never answer the original request. Only decide CLEAR or ask questions.
4. Be strict. Prefer asking for clarification when in doubt.
5. Keep clarifying questions short and focused.

User request:
{user_message}

Your response:"""


def check_ambiguity(user_message: str) -> str:
    """
    Returns either "CLEAR" or clarifying questions.
    """
    prompt = AMBIGUITY_PROMPT.format(user_message=user_message)
    
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    response = call_llm(messages, temperature=0.1)
    if response:
        clean_op= response.strip()
    return clean_op