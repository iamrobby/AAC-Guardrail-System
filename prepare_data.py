import json
import random

with open("data/sharded_instructions_600.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Keep only code and math
filtered = [item for item in data if item["task"] in ["code", "math"]]
print(f"Found {len(filtered)} code + math examples")

random.seed(42)
subset = random.sample(filtered, min(25, len(filtered)))  # 25 examples is good for start

eval_data = []

for idx, item in enumerate(subset, 1):
    shards = item["shards"]
    
    first_shard = shards[0]["shard"]
    remaining_shards = [s["shard"] for s in shards[1:]]
    full_instruction = " ".join([s["shard"] for s in shards])
    
    eval_data.append({
        "id": idx,
        "task": item["task"],
        "task_id": item.get("task_id", ""),
        "incomplete_prompt": first_shard,
        "remaining_shards": remaining_shards,
        "full_instruction": full_instruction,
        "num_shards": len(shards)
    })

with open("data/full_eval_set.json", "w", encoding="utf-8") as f:
    json.dump(eval_data, f, indent=2, ensure_ascii=False)

print(f"\nCreated data/full_eval_set.json with {len(eval_data)} examples")
print("Each example contains:")
print(" - incomplete_prompt (first shard)")
print(" - remaining_shards (to simulate user answers)")
print(" - full_instruction")