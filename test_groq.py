from src.llm_client import call_llm

messages = [
    {"role": "user", "content": "Say 'Groq is working' and nothing else."}
]

print(call_llm(messages))