# run_tools.py
# Part C, Evaluation 2: generate + tool calling.
#
# The model gets one tool, check_citations, which checks whether citations really appear in the judgment excerpt. 
# The model is asked to check its citations with the tool before giving its final answer, and to keep only the ones the tool confirms.


import json
import os
import re
import time

import torch
from hydra import compose, initialize
from src.generation import GenerationManager
from src.schema import register_configs
from src.tools import register_tool

# ---- change these for each run ----
RUN_NAME = "custom2_tools"
INPUT_FILE = "data/ip_corpus/eval_inputs.jsonl"
TEMPERATURE = 0.2
TOP_P = 0.9
MAX_NEW_TOKENS = 768
NUM_RECORDS = 50        # set to 2 for a quick test
# -----------------------------------


# ---------- the tool ----------

# what the model is shown about the tool
TOOLS = [{
    "name": "check_citations",
    "description": "Check whether case citations really appear in the judgment excerpt. "
                   "Returns true for each citation found in the excerpt and false otherwise.",
    "parameters": {
        "type": "object",
        "properties": {
            "citations": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Case citations to check, e.g. \"[2008] FCA 803\" or \"(2003) 59 IPR 191\"",
            }
        },
        "required": ["citations"],
    },
}]

CITATION = r"\[\d{4}\] [A-Z][A-Za-z]* \d+|\(\d{4}\) \d+ [A-Z][A-Za-z]* \d+"
current_excerpt = ""   # set to each record's text before the model runs


@register_tool("check_citations")   # LLMBox runs this when the model calls the tool
def check_citations(citations):
    if isinstance(citations, str):   # in case the model passes one string instead of a list
        citations = [citations]
    results = {}
    for item in citations:
        # if the model passes "Case name [2008] FCA 803", check just the citation part
        found = re.findall(CITATION, item)
        to_check = found[0] if found else item.strip()
        results[item] = to_check in current_excerpt
    return results


# ---------- the prompt ----------

SYSTEM_PROMPT = (
    "You are a legal research assistant at an Australian intellectual property "
    "law firm. You read excerpts of Federal Court of Australia judgments and "
    "extract information for a junior lawyer's precedent research. Report only "
    "what the excerpt itself states."
)

USER_PROMPT = """Read the judgment excerpt below.

Step 1: Call the check_citations tool with every case citation you think the excerpt cites.
Step 2: After you get the tool results, give your final answer. In CASES_CITED, include only citations the tool returned true for.

Final answer format:
IP_AREAS: <comma-separated, choose from: copyright, patent, trade_mark, passing_off, registered_design>
CASES_CITED: <semicolon-separated list of the cases cited in the excerpt, each with its citation, or NONE>
LEGISLATION_CITED: <semicolon-separated list of the Acts or sections cited in the excerpt, or NONE>

Excerpt:
<<<
{text}
>>>"""


# ---------- load LLMBox and the model ----------

register_configs()
with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config", overrides=[
        "model=phi4_instruct",
        "mode=tool_calling",
        "tool_calling.enabled=true",
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


def count_tokens(messages):
    ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    return len(ids)


# ---------- run ----------

results = []
for i, record in enumerate(records):
    current_excerpt = record["raw_text"]

    # the same system prompt LLMBox builds in mode=tool_calling for Phi-4-mini
    system = llmbox.build_functools_system_prompt(SYSTEM_PROMPT, TOOLS)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": USER_PROMPT.format(text=record["raw_text"])},
    ]
    first_prompt_tokens = count_tokens(messages)

    torch.manual_seed(42)
    start = time.time()
    answer, tool_calls = llmbox.run_generic_tool_turn(
        model, tokenizer, device, messages, cfg, TOOLS, cfg.tool_calling.tool_choice)
    seconds = time.time() - start

    # if the tool was used, the model ran twice: count both prompts and both replies
    if tool_calls:
        input_tokens = first_prompt_tokens + count_tokens(messages[:-1])
        output_tokens = (len(tokenizer.encode(messages[2]["content"], add_special_tokens=False))
                         + len(tokenizer.encode(answer, add_special_tokens=False)))
    else:
        input_tokens = first_prompt_tokens
        output_tokens = len(tokenizer.encode(answer, add_special_tokens=False))

    results.append({
        "doc_id": record["doc_id"],
        "answer": answer,
        "tool_called": bool(tool_calls),
        "tool_calls": tool_calls,
        "seconds": round(seconds, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })
    print(f"{i + 1}/{len(records)}  {record['doc_id']}  {seconds:.1f}s  "
          f"{output_tokens} tokens  tool {'used' if tool_calls else 'NOT used'}")


# ---------- save and summarise ----------

os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{RUN_NAME}.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

n = len(results)
rejected = sum(1 for r in results for c in r["tool_calls"] for ok in c["result"].values() if not ok)
print("\nRecords where the model used the tool:", sum(r["tool_called"] for r in results), "/", n)
print("Citations the tool rejected (not in the excerpt):", rejected)
print("Saved to", f"outputs/{RUN_NAME}.jsonl")
