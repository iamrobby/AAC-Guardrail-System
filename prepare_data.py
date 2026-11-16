"""Build the evaluation sets from the sharded-instruction benchmark.

Creates:
  data/full_eval_set.json     - end-to-end set (first shard + hidden remaining shards + gold answer/tests)
  data/gate_eval_sharded.json - gate set: first shard = AMBIGUOUS, full instruction = CLEAR

Only auto-gradable tasks are kept: GSM8K math (numeric answer) and HumanEval code (unit tests).
Note: re-running this overwrites the two output files above.
"""
import argparse
import json
import random
from pathlib import Path


def parse_gsm8k_answer(answer: str) -> str:
    final = answer.split("####")[-1].strip().replace(",", "")
    float(final)  # fail loudly if the gold answer is not numeric
    return final


def build_item(idx: int, raw: dict) -> dict:
    shards = [s["shard"].strip() for s in raw["shards"]]
    item = {
        "id": idx,
        "task": raw["task"],
        "task_id": raw.get("task_id", ""),
        "incomplete_prompt": shards[0],
        "remaining_shards": shards[1:],
        "full_instruction": " ".join(shards),
        "num_shards": len(shards),
    }
    if raw["task"] == "math":
        item["gold_answer"] = parse_gsm8k_answer(raw["answer"])
    else:
        item["func_name"] = raw["metadata"]["func_name"]
        item["test"] = raw["test"]
    return item


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/sharded_instructions_600.json")
    parser.add_argument("--n", type=int, default=30, help="total examples (split evenly math/code)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    with open(args.src, "r", encoding="utf-8") as f:
        data = json.load(f)

    math_pool = [x for x in data if x["task"] == "math" and "####" in x.get("answer", "")]
    code_pool = [x for x in data if x["task"] == "code" and x.get("source") == "humaneval" and "test" in x]
    print(f"Gradable pool: {len(math_pool)} math, {len(code_pool)} code (HumanEval)")

    rng = random.Random(args.seed)
    n_math = args.n // 2
    n_code = args.n - n_math
    chosen = rng.sample(math_pool, min(n_math, len(math_pool))) + rng.sample(code_pool, min(n_code, len(code_pool)))

    items = [build_item(i, raw) for i, raw in enumerate(chosen, 1)]

    gate_cases = []
    for item in items:
        gate_cases.append({"id": item["id"], "task": item["task"], "message": item["incomplete_prompt"],
                           "label": "AMBIGUOUS", "type": "first_shard"})
        gate_cases.append({"id": 100 + item["id"], "task": item["task"], "message": item["full_instruction"],
                           "label": "CLEAR", "type": "full"})

    Path("data").mkdir(exist_ok=True)
    with open("data/full_eval_set.json", "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)
    with open("data/gate_eval_sharded.json", "w", encoding="utf-8") as f:
        json.dump(gate_cases, f, indent=2, ensure_ascii=False)

    print(f"Wrote data/full_eval_set.json ({len(items)} items) and data/gate_eval_sharded.json ({len(gate_cases)} cases)")


if __name__ == "__main__":
    main()
