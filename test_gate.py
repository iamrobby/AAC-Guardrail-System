"""Quick manual smoke test of the gate (needs GROQ_API_KEY).   python test_gate.py"""
from src.ambiguity_gate import check_ambiguity

tests = [
    "Write a Python function to calculate factorial",
    "Write a Python function",
    "Solve this math problem: A train leaves at 3pm",
    "What is 15 * 23?",
    "Create a login page",
    "Help me with my code",
]

print("=== Ambiguity Gate Tests ===\n")
for i, msg in enumerate(tests, 1):
    result = check_ambiguity(msg)
    print(f"Test {i}: {msg}")
    print(f"Result -> {result.label} (score {result.ambiguity_score:.2f})")
    if result.questions:
        print("Questions:", *result.questions, sep="\n  - ")
    if result.error:
        print("ERROR:", result.error)
    print("-" * 50)
