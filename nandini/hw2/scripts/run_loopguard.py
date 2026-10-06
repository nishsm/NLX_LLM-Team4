# run_loopguard.py
# Part C, Evaluation 4 (custom feature 2): a loop guard.
#
# Problem it fixes: at low temperature, about 1 in 10 answers got stuck repeating itself until it hit the token limit. Those answers are useless and take 5-8x longer.
#
# What it does: after each answer, check whether it looped. If it did, generate the answer again once with a repetition penalty
# makes the model less likely to repeat words it has already written.
#
# The prompt and settings are identical to the guardrail run, and the seed is set the same way

import json
import os
import time

import torch
from hydra import compose, initialize
from src.generation import GenerationManager
from src.schema import register_configs

# change these for each run 
RUN_NAME = "custom4_loopguard"
INPUT_FILE = "data/ip_corpus/eval_inputs.jsonl"
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_NEW_TOKENS = 768
NUM_RECORDS = 50          # set to e.g. 2 for a quick test
RETRY_PENALTY = 1.15      # repetition penalty used only for the retry (1.0 = none)


SYSTEM_PROMPT = (
    "You are a legal research assistant at an Australian intellectual property "
    "law firm. You read excerpts of Federal Court of Australia judgments and "
    "extract information for a junior lawyer's precedent research. Report only "
    "what the excerpt itself states."
)

INSTRUCTION = """Read the judgment excerpt below and answer in exactly this format:

IP_AREAS: <comma-separated, choose from: copyright, patent, trade_mark, passing_off, registered_design>
CASES_CITED: <semicolon-separated list of the cases cited in the excerpt, each with its citation, or NONE>
LEGISLATION_CITED: <semicolon-separated list of the Acts or sections cited in the excerpt, or NONE>"""


def looped(answer, output_tokens):
    """An answer looped if it ran into the token limit, or if any line
    (or any item in a list) appears 3 or more times."""
    if output_tokens >= MAX_NEW_TOKENS - 2:
        return True
    pieces = [p.strip() for p in answer.replace("\n", ";").split(";") if p.strip()]
    return any(pieces.count(p) >= 3 for p in set(pieces))


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

llmbox = GenerationManager()
model, tokenizer, device = llmbox._load_model_and_tokenizer(cfg)

records = []
with open(INPUT_FILE) as f:
    for line in f:
        records.append(json.loads(line))
records = records[:NUM_RECORDS]

results = []
for i, record in enumerate(records):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": INSTRUCTION + "\n\nExcerpt:\n<<<\n" + record["raw_text"] + "\n>>>"},
    ]
    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    prompt_tokens = len(prompt_ids)

    start = time.time()
    torch.manual_seed(42)
    answer = llmbox._generate_once(model, tokenizer, device, messages, cfg)
    out_tokens = len(tokenizer.encode(answer, add_special_tokens=False))

    retried = False
    first_answer_looped = looped(answer, out_tokens)
    input_tokens, output_tokens = prompt_tokens, out_tokens

    if first_answer_looped:
        retried = True
        cfg.generation.repetition_penalty = RETRY_PENALTY
        torch.manual_seed(42)
        answer = llmbox._generate_once(model, tokenizer, device, messages, cfg)
        cfg.generation.repetition_penalty = 1.0
        retry_tokens = len(tokenizer.encode(answer, add_special_tokens=False))
        input_tokens += prompt_tokens       # the prompt was processed twice
        output_tokens += retry_tokens       # count both answers in the cost
        out_tokens = retry_tokens

    seconds = time.time() - start

    results.append({
        "doc_id": record["doc_id"],
        "answer": answer,
        "first_answer_looped": first_answer_looped,
        "retried": retried,
        "still_looped": retried and looped(answer, out_tokens),
        "seconds": round(seconds, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "final_answer_tokens": out_tokens,
    })
    print(f"{i + 1}/{len(records)}  {record['doc_id']}  {seconds:.1f}s  "
          f"{out_tokens} tokens{'  (looped -> retried)' if retried else ''}")

os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{RUN_NAME}.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

print("\nAnswers that looped the first time:", sum(r["first_answer_looped"] for r in results), "/", len(results))
print("Still looped after the retry:", sum(r["still_looped"] for r in results))
print("Saved to", f"outputs/{RUN_NAME}.jsonl")
