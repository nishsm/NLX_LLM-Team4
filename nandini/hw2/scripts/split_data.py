# split_data.py
# Splits the corpus into a development set (Part B) and an evaluation set (Part C).
# All records from the same case go to the same side, so no judgment appears in both.

import json
import random

# load all 218 records
records = []
with open("data/ip_corpus/corpus.jsonl") as f:
    for line in f:
        records.append(json.loads(line))

# get the list of 50 case ids and shuffle them (seed 42 so it's the same every time)
case_ids = sorted(set(r["metadata"]["austlii_id"] for r in records))
random.seed(42)
random.shuffle(case_ids)

dev_cases = case_ids[:25]
eval_cases = case_ids[25:]

# keep only judgment sections (the records that have text to send to the model)
sections = [r for r in records if r["metadata"]["document_type"] == "judgment_section"]

dev = [r for r in sections if r["metadata"]["austlii_id"] in dev_cases]
evaluation = [r for r in sections if r["metadata"]["austlii_id"] in eval_cases]

# each side has 75 sections, shuffle and take 50 for the runs
random.shuffle(dev)
random.shuffle(evaluation)

def save(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

save(dev[:50], "data/ip_corpus/dev_inputs.jsonl")
save(dev[50:], "data/ip_corpus/dev_spare.jsonl")
save(evaluation[:50], "data/ip_corpus/eval_inputs.jsonl")
save(evaluation[50:], "data/ip_corpus/eval_spare.jsonl")

print("dev cases:", len(dev_cases), "| dev sections:", len(dev))
print("eval cases:", len(eval_cases), "| eval sections:", len(evaluation))
