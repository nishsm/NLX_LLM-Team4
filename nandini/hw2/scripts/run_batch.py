# run_batch.py
# Sends 50 records through LLMBox's generate mode and saves each answer, how long it took, and how many tokens it used.

import json
import os
import time
import torch
from hydra import compose, initialize
from src.generation import GenerationManager
from src.schema import register_configs

# change these for each run 
RUN_NAME = "baseline3"
INPUT_FILE = "data/ip_corpus/dev_inputs.jsonl"
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_NEW_TOKENS = 768
NUM_RECORDS = 50   # set to small number e.g. 2 for a quick test


SYSTEM_PROMPT = (
    "You are a legal research assistant at an Australian intellectual property "
    "law firm. You read excerpts of Federal Court of Australia judgments and "
    "extract information for a junior lawyer's precedent research. Report only "
    "what the excerpt itself states."
)

USER_PROMPT = """Read the judgment excerpt below and answer in exactly this format:

IP_AREAS: <comma-separated, choose from: copyright, patent, trade_mark, passing_off, registered_design>
CASES_CITED: <semicolon-separated list of the cases cited in the excerpt, each with its citation, or NONE>
LEGISLATION_CITED: <semicolon-separated list of the Acts or sections cited in the excerpt, or NONE>

Excerpt:
<<<
{text}
>>>"""

# load the same settings LLMBox uses on the command line
register_configs()
with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config", overrides=[
        "model=phi4_instruct",
        "mode=generate",
        "model.source=local",
        "model.local_path=./models/llms/microsoft/Phi-4-mini-instruct",
        f"generation.temperature={TEMPERATURE}",
        f"generation.top_p={TOP_P}",
        f"generation.max_new_tokens={MAX_NEW_TOKENS}",
    ])

# load the model once
llmbox = GenerationManager()
model, tokenizer, device = llmbox._load_model_and_tokenizer(cfg)

# load the 50 records
records = []
with open(INPUT_FILE) as f:
    for line in f:
        records.append(json.loads(line))
records = records[:NUM_RECORDS]

results = []
for i, record in enumerate(records):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(text=record["raw_text"])},
    ]

    # LLMBox doesn't set the seed itself, so I do it here to make runs repeatable
    torch.manual_seed(42)

    start = time.time()
    answer = llmbox._generate_once(model, tokenizer, device, messages, cfg)
    seconds = time.time() - start

    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    input_tokens = len(prompt_ids)
    output_tokens = len(tokenizer.encode(answer, add_special_tokens=False))

    results.append({
        "doc_id": record["doc_id"],
        "answer": answer,
        "seconds": round(seconds, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })
    print(f"{i + 1}/{len(records)}  {record['doc_id']}  {seconds:.1f}s  {output_tokens} tokens")

# save everything
os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{RUN_NAME}.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("Average seconds per record:", round(sum(r["seconds"] for r in results) / len(results), 2))
print("Saved to", f"outputs/{RUN_NAME}.jsonl")
