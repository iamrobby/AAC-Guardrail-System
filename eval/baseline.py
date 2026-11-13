import json
import os
from tqdm import tqdm
from src.llm_client import call_llm

# Load evaluation data
with open("data/full_eval_set.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

print(f"Running Baseline on {len(data)} examples...\n")

for item in tqdm(data):
    incomplete_prompt = item["incomplete_prompt"]
    
    # Force the model to answer immediately
    messages = [
        {
            "role": "system",
            "content": "You are a helpful assistant. Answer the user's request directly and completely."
        },
        {
            "role": "user",
            "content": incomplete_prompt
        }
    ]
    
    try:
        answer = call_llm(messages, temperature=0.2)
    except Exception as e:
        answer = f"ERROR: {str(e)}"
    
    results.append({
        "id": item["id"],
        "task": item["task"],
        "incomplete_prompt": incomplete_prompt,
        "baseline_answer": answer,
        "full_instruction": item["full_instruction"]  # for later comparison
    })

# Save results
os.makedirs("results", exist_ok=True)

with open("results/baseline_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("\nBaseline finished!")
print(f"Results saved to: results/baseline_results.json")