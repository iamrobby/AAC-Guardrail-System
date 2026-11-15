"""End-to-end comparison on data/full_eval_set.json.

Modes:
  baseline - model answers the incomplete first shard immediately
  aac      - guardrail: gate -> clarifying questions -> simulated user -> answer
  oracle   - model answers the full instruction (upper bound)

Run from the project root:
  python -m eval.run_eval                       # all three modes
  python -m eval.run_eval --modes baseline aac --limit 5
"""
import argparse
import json
from collections import defaultdict
from typing import Callable, Dict, List

from tqdm import tqdm

from src import config
from src.llm_client import LLMError
from src.pipeline import AACGuardrail, answer_request
from src.user_simulator import simulate_user_reply
from eval.grader import format_hint, grade_item


def run_baseline(item: dict) -> dict:
    return {"answer": answer_request(item["incomplete_prompt"], None, format_hint(item))}


def run_oracle(item: dict) -> dict:
    return {"answer": answer_request(item["full_instruction"], None, format_hint(item))}


def make_aac_runner(guard: AACGuardrail) -> Callable[[dict], dict]:
    def run_aac(item: dict) -> dict:
        def reply_fn(questions: List[str]) -> str:
            return simulate_user_reply(item["incomplete_prompt"], questions, item["remaining_shards"])

        result = guard.run(item["incomplete_prompt"], reply_fn, format_hint(item))
        return {
            "answer": result.answer,
            "rounds_asked": result.rounds_asked,
            "history": result.history,
            "gate_scores": [g.ambiguity_score for g in result.gates],
            "gate_errors": sum(1 for g in result.gates if g.error),
        }
    return run_aac


def evaluate(mode: str, runner: Callable[[dict], dict], items: List[dict]) -> List[dict]:
    rows = []
    for item in tqdm(items, desc=mode):
        row = {"id": item["id"], "task": item["task"], "prompt": item["incomplete_prompt"]}
        try:
            row.update(runner(item))
            row["correct"] = grade_item(item, row["answer"])
            row["error"] = None
        except LLMError as exc:
            row.update({"answer": None, "correct": False, "error": str(exc)})
        rows.append(row)
    return rows


def summarise(rows: List[dict]) -> Dict:
    ok = [r for r in rows if not r["error"]]
    by_task = defaultdict(lambda: [0, 0])
    for r in ok:
        by_task[r["task"]][1] += 1
        by_task[r["task"]][0] += r["correct"]
    summary = {
        "n": len(rows),
        "errors": len(rows) - len(ok),
        "accuracy": sum(r["correct"] for r in ok) / len(ok) if ok else 0.0,
        "by_task": {t: c / n for t, (c, n) in by_task.items()},
    }
    if ok and "rounds_asked" in ok[0]:
        summary["clarification_rate"] = sum(r["rounds_asked"] > 0 for r in ok) / len(ok)
        summary["avg_rounds"] = sum(r["rounds_asked"] for r in ok) / len(ok)
        summary["gate_errors"] = sum(r["gate_errors"] for r in ok)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/full_eval_set.json")
    parser.add_argument("--modes", nargs="+", default=["baseline", "aac", "oracle"],
                        choices=["baseline", "aac", "oracle"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=config.AMBIGUITY_THRESHOLD)
    parser.add_argument("--max-rounds", type=int, default=config.MAX_CLARIFY_ROUNDS)
    args = parser.parse_args()

    with open(args.data, "r", encoding="utf-8") as f:
        items = json.load(f)
    if args.limit:
        items = items[: args.limit]

    runners = {
        "baseline": run_baseline,
        "aac": make_aac_runner(AACGuardrail(args.threshold, args.max_rounds)),
        "oracle": run_oracle,
    }
    config.RESULTS_DIR.mkdir(exist_ok=True)
    summaries = {}
    for mode in args.modes:
        rows = evaluate(mode, runners[mode], items)
        summaries[mode] = summarise(rows)
        with open(config.RESULTS_DIR / f"{mode}_results.json", "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, ensure_ascii=False)

    with open(config.RESULTS_DIR / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    print(f"\n{'mode':10} {'acc':>6} {'math':>6} {'code':>6} {'asked':>6} {'errors':>7}")
    for mode, s in summaries.items():
        asked = f"{s['clarification_rate']:.0%}" if "clarification_rate" in s else "-"
        print(f"{mode:10} {s['accuracy']:6.1%} {s['by_task'].get('math', 0):6.1%} "
              f"{s['by_task'].get('code', 0):6.1%} {asked:>6} {s['errors']:>7}")
    if any(s["errors"] for s in summaries.values()):
        print("\nWARNING: some items failed with LLM errors and are excluded from accuracy.")


if __name__ == "__main__":
    main()
