"""Interactive chat with the AAC guardrail.   python main.py [--debug]"""
import argparse
from typing import List

from src.llm_client import LLMError
from src.pipeline import AACGuardrail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="show the gate's decision")
    args = parser.parse_args()

    guard = AACGuardrail()
    print("AAC guardrail chat. Type 'exit' to quit.\n")

    def ask_user(questions: List[str]) -> str:
        print("\nAssistant: Before I answer, I need a bit more information:")
        for i, question in enumerate(questions, 1):
            print(f"  {i}. {question}")
        return input("You: ").strip()

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if message.lower() in {"exit", "quit"}:
            break
        if not message:
            continue
        try:
            result = guard.run(message, reply_fn=ask_user)
        except LLMError as exc:
            print(f"\n[error] {exc}\n")
            continue
        if args.debug:
            for i, gate in enumerate(result.gates):
                print(f"[gate {i}] {gate.label} score={gate.ambiguity_score:.2f} "
                      f"categories={gate.categories} error={gate.error}")
        print(f"\nAssistant: {result.answer}\n")


if __name__ == "__main__":
    main()
