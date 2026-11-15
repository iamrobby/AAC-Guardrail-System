"""Evaluate the ambiguity gate on labelled CLEAR / AMBIGUOUS messages.

AMBIGUOUS is the positive class: the guardrail's job is to catch ambiguous requests.
  recall    = share of truly ambiguous requests that were caught
  precision = share of flagged requests that were truly ambiguous
  specificity = share of clear requests that were let through (1 - annoyance rate)

Usage:
  python eval_gate.py                                   # data/gate_eval.json
  python eval_gate.py data/gate_eval.json data/gate_eval_lost.json --sweep
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from src import config
from src.ambiguity_gate import AMBIGUOUS, check_ambiguity


def group_of(case: dict) -> str:
    return case.get("domain") or case.get("task") or "all"


def compute_metrics(rows: List[dict], threshold: float) -> Dict[str, float]:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted_ambiguous = row["score"] >= threshold
        truly_ambiguous = row["expected"] == AMBIGUOUS
        if truly_ambiguous and predicted_ambiguous:
            tp += 1
        elif truly_ambiguous:
            fn += 1
        elif predicted_ambiguous:
            fp += 1
        else:
            tn += 1

    def ratio(a: int, b: int) -> float:
        return a / b if b else 0.0

    precision, recall = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": ratio(tp + tn, tp + tn + fp + fn),
        "precision": precision,
        "recall": recall,
        "specificity": ratio(tn, tn + fp),
        "f1": ratio(2 * precision * recall, precision + recall),
    }


def run_gate(cases: List[dict], threshold: float) -> List[dict]:
    rows = []
    for case in tqdm(cases, desc="gate", leave=False):
        result = check_ambiguity(case["message"], threshold=threshold)
        rows.append({
            "id": case["id"],
            "group": group_of(case),
            "message": case["message"],
            "expected": case["label"],
            "predicted": result.label,
            "score": result.ambiguity_score,
            "questions": result.questions,
            "error": result.error,
            "raw": result.raw,
        })
    return rows


def report(name: str, rows: List[dict], threshold: float, sweep: bool) -> dict:
    m = compute_metrics(rows, threshold)
    errors = sum(1 for r in rows if r["error"])
    print(f"\n{'=' * 60}\n{name}  (threshold={threshold}, n={len(rows)})\n{'=' * 60}")
    print(f"Confusion (positive = AMBIGUOUS): TP={m['tp']} FP={m['fp']} TN={m['tn']} FN={m['fn']}")
    print(f"Accuracy {m['accuracy']:.3f} | Precision {m['precision']:.3f} | Recall {m['recall']:.3f} "
          f"| Specificity {m['specificity']:.3f} | F1 {m['f1']:.3f}")
    if errors:
        print(f"WARNING: {errors} gate call(s) failed and were counted with the fail-safe label.")

    by_group = defaultdict(lambda: [0, 0])
    for r in rows:
        by_group[r["group"]][1] += 1
        by_group[r["group"]][0] += r["predicted"] == r["expected"]
    print("By group: " + ", ".join(f"{g}: {c}/{t}" for g, (c, t) in sorted(by_group.items())))

    mistakes = [r for r in rows if r["predicted"] != r["expected"]]
    if mistakes:
        print("\nMistakes:")
        for r in mistakes:
            print(f"  ID {r['id']} [{r['group']}] expected {r['expected']}, got {r['predicted']} "
                  f"(score {r['score']:.2f}): {r['message'][:90]}")

    sweep_table = []
    if sweep:
        print("\nThreshold sweep:\n  thr   acc  prec  rec   f1")
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            s = compute_metrics(rows, t)
            sweep_table.append({"threshold": t, **s})
            print(f"  {t:.1f}  {s['accuracy']:.2f}  {s['precision']:.2f}  {s['recall']:.2f}  {s['f1']:.2f}")
    return {"metrics": m, "errors": errors, "sweep": sweep_table, "rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="*", default=["data/gate_eval.json"])
    parser.add_argument("--threshold", type=float, default=config.AMBIGUITY_THRESHOLD)
    parser.add_argument("--sweep", action="store_true", help="also print metrics for thresholds 0.1-0.9")
    args = parser.parse_args()

    config.RESULTS_DIR.mkdir(exist_ok=True)
    for path in args.datasets:
        with open(path, "r", encoding="utf-8") as f:
            cases = json.load(f)
        rows = run_gate(cases, args.threshold)
        out = report(Path(path).name, rows, args.threshold, args.sweep)
        out_path = config.RESULTS_DIR / f"gate_eval_{Path(path).stem}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
