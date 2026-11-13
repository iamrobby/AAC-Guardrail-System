# import json
# from src.ambiguity_gate import check_ambiguity
# with open("data/gate_eval.json","r") as file:
#     tests=json.load(file)
    
# def is_clear(response: str) -> bool:
#     """Simple rule: if response is exactly 'CLEAR' → True, else False"""
#     return response.strip().upper() == "CLEAR"
# tp = tn = fp = fn = 0

# print("=== Evaluation Results ===\n")

# for item in tests:
#     message = item["message"]
#     true_label = item["label"]          # "CLEAR" or "AMBIGUOUS"
    
#     prediction = check_ambiguity(message)
#     pred_is_clear = is_clear(prediction)
    
#     # Update confusion matrix
#     if true_label == "CLEAR" and pred_is_clear:
#         tp += 1
#     elif true_label == "AMBIGUOUS" and not pred_is_clear:
#         tn += 1
#     elif true_label == "AMBIGUOUS" and pred_is_clear:
#         fp += 1
#     elif true_label == "CLEAR" and not pred_is_clear:
#         fn += 1
    
#     # Optional: print mistakes
#     if (true_label == "CLEAR") != pred_is_clear:
#         print(f"Mismatch | ID: {item['id']}")
#         print(f"Message : {message}")
#         print(f"True    : {true_label}")
#         print(f"Pred    : {prediction[:100]}...")
#         print("-" * 50)

# # Calculate metrics
# total = tp + tn + fp + fn
# accuracy = (tp + tn) / total if total > 0 else 0
# precision = tp / (tp + fp) if (tp + fp) > 0 else 0
# recall = tp / (tp + fn) if (tp + fn) > 0 else 0
# f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# print("\n=== Final Metrics ===")
# print(f"True Positives  (TP): {tp}")
# print(f"True Negatives  (TN): {tn}")
# print(f"False Positives (FP): {fp}")
# print(f"False Negatives (FN): {fn}")
# print()
# print(f"Accuracy  : {accuracy:.3f}")
# print(f"Precision : {precision:.3f}")
# print(f"Recall    : {recall:.3f}")
# print(f"F1-Score  : {f1:.3f}")
import json
from src.ambiguity_gate import check_ambiguity

def evaluate_gate(json_path: str):
    with open(json_path, "r") as f:
        test_cases = json.load(f)

    results = []
    correct = 0

    for case in test_cases:
        msg_id    = case["id"]
        message   = case["message"]
        expected  = case["label"]        # "CLEAR" or "AMBIGUOUS"
        domain    = case["domain"]

        # Run the gate
        response = check_ambiguity(message)

        # Predict: if response is exactly "CLEAR" → CLEAR, else → AMBIGUOUS
        predicted = "CLEAR" if response.strip().upper() == "CLEAR" else "AMBIGUOUS"
        is_correct = predicted == expected

        if is_correct:
            correct += 1

        results.append({
            "id":        msg_id,
            "domain":    domain,
            "message":   message,
            "expected":  expected,
            "predicted": predicted,
            "response":  response,
            "correct":   is_correct
        })

        # Print live progress
        status = "✅" if is_correct else "❌"
        print(f"{status} ID {msg_id:02d} | {domain:6} | Expected: {expected:9} | Got: {predicted}")

    accuracy = correct / len(test_cases) * 100
    print(f"\n{'='*50}")
    print(f"Gate Accuracy: {correct}/{len(test_cases)} = {accuracy:.1f}%")

    # Break down by domain
    for domain in ["math", "code"]:
        domain_cases = [r for r in results if r["domain"] == domain]
        domain_correct = sum(1 for r in domain_cases if r["correct"])
        print(f"  {domain}: {domain_correct}/{len(domain_cases)}")

    # Show failures
    failures = [r for r in results if not r["correct"]]
    if failures:
        print(f"\n❌ Failures:")
        for f in failures:
            print(f"  ID {f['id']}: '{f['message']}'")
            print(f"    Expected {f['expected']}, got {f['predicted']}")
            print(f"    Model said: '{f['response'][:80]}'")

    return results

if __name__ == "__main__":
    evaluate_gate("data/gate_eval.json")