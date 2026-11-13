from src.ambiguity_gate import check_ambiguity
tests = [
    "Write a Python function to calculate factorial",
    "Write a Python function",
    "Solve this math problem: A train leaves at 3pm",
    "What is 15 * 23?",
    "Create a login page",
    "Help me with my code"
]

print("=== Ambiguity Gate Tests ===\n")

for i, msg in enumerate(tests, 1):
    print(f"Test {i}: {msg}")
    result = check_ambiguity(msg)
    print(f"Result → {result}")
    print("-" * 50)