# run_structured.py
# Part C, Evaluation 1: generate + structured output.
#
# 1. A Pydantic model defines what a correct answer looks like.
# 2. It is turned into a JSON schema and added to the system prompt, exactly the way LLMBox's structured_output mode does it.
# 3. Each answer is checked against the Pydantic model.
# 4. Valid answers are also written in the same "IP_AREAS: ..." format as Part B, so score.py can score them and the results are comparable.

import json
import os
import time
from typing import List, Literal

import torch
from hydra import compose, initialize
from pydantic import BaseModel, Field, ValidationError
from src.generation import GenerationManager
from src.schema import register_configs

# change these for each run:
RUN_NAME = "custom1_structured"
INPUT_FILE = "data/ip_corpus/eval_inputs.jsonl"
TEMPERATURE = 0.2       # whichever settings did better in Part B
TOP_P = 0.9
MAX_NEW_TOKENS = 768
NUM_RECORDS = 50        # set to e.g. 2 for a quick test
SCHEMA_VERSION = "optional"   # "optional" or "required" (see the two classes below)



# the Pydantic models:

IPArea = Literal["copyright", "patent", "trade_mark", "passing_off", "registered_design"]


class CaseCitation(BaseModel):
    case_name: str
    citation: str


# Lists may be empty: the model is allowed to say "nothing cited".
class ExtractionOptional(BaseModel):
    primary_ip_area: IPArea
    secondary_ip_areas: List[IPArea] = []
    cases_cited: List[CaseCitation] = []
    legislation_cited: List[str] = []


# Lists must have at least one item: the model MUST name a case and an Act, even when the excerpt cites none. 
# Used to test whether required fields push the model into making things up.
class ExtractionRequired(BaseModel):
    primary_ip_area: IPArea
    secondary_ip_areas: List[IPArea] = []
    cases_cited: List[CaseCitation] = Field(min_length=1)
    legislation_cited: List[str] = Field(min_length=1)


Schema = ExtractionOptional if SCHEMA_VERSION == "optional" else ExtractionRequired

# save the schema to a file, so it can also be used with startllm.py and put in the memo
os.makedirs("schemas", exist_ok=True)
schema_text = json.dumps(Schema.model_json_schema(), indent=2)
with open(f"schemas/ip_extraction_{SCHEMA_VERSION}.json", "w") as f:
    f.write(schema_text)


# prompt: 

SYSTEM_PROMPT = (
    "You are a legal research assistant at an Australian intellectual property "
    "law firm. You read excerpts of Federal Court of Australia judgments and "
    "extract information for a junior lawyer's precedent research. Report only "
    "what the excerpt itself states."
)

SCHEMA_INSTRUCTION = (
    "\n\nRespond with ONLY a single JSON object that strictly matches "
    f"this JSON Schema, with no other text:\n{schema_text}"
)

USER_PROMPT = """Extract the IP area(s), the cases cited and the legislation cited from this judgment excerpt.

Rules:
- Do NOT repeat the schema. Return one JSON object with your answers filled in.
- primary_ip_area is the single main area of IP law in the excerpt.
- If no cases or no legislation are cited in the excerpt, use an empty list [].
- Never write placeholders such as "Not provided". Leave the list empty instead.
- Each item in legislation_cited is a plain string.

Example of the answer format only (do not copy these values):
{{"primary_ip_area": "trade_mark", "secondary_ip_areas": ["passing_off"], "cases_cited": [{{"case_name": "Example Pty Ltd v Sample Ltd", "citation": "[1999] FCA 1"}}], "legislation_cited": ["Example Act 1990 (Cth)"]}}

Excerpt:
<<<
{text}
>>>"""


# load LLMBox and the model:

register_configs()
with initialize(version_base=None, config_path="conf"):
    cfg = compose(config_name="config", overrides=[
        "model=phi4_instruct",
        "mode=structured_output",
        "structured_output.enabled=true",
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


# run: 

results = []
for i, record in enumerate(records):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + SCHEMA_INSTRUCTION},
        {"role": "user", "content": USER_PROMPT.format(text=record["raw_text"])},
    ]

    torch.manual_seed(42)
    start = time.time()
    raw_answer = llmbox._generate_once(model, tokenizer, device, messages, cfg)
    seconds = time.time() - start

    prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
    input_tokens = len(prompt_ids)
    output_tokens = len(tokenizer.encode(raw_answer, add_special_tokens=False))

    # Check 1: is it valid JSON exactly as returned? (LLMBox's own "strict" check)
    try:
        json.loads(raw_answer)
        json_as_returned = True
    except json.JSONDecodeError:
        json_as_returned = False

    # Check 2: after removing ```json fences, if it matches the Pydantic model
    cleaned = raw_answer.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    try:
        parsed = Schema.model_validate_json(cleaned)
        schema_valid = True
        error = None
    except ValidationError as e:
        parsed = None
        schema_valid = False
        error = str(e).splitlines()[0:3]

    # write valid answers in the Part B text format so score.py can score them
    if parsed:
        ip_areas = [parsed.primary_ip_area] + parsed.secondary_ip_areas
        cases = [c.case_name + " " + c.citation for c in parsed.cases_cited]
        answer = (
            "IP_AREAS: " + ", ".join(ip_areas) + "\n"
            "CASES_CITED: " + ("; ".join(cases) or "NONE") + "\n"
            "LEGISLATION_CITED: " + ("; ".join(parsed.legislation_cited) or "NONE")
        )
        primary = parsed.primary_ip_area
    else:
        answer = ""
        primary = None

    results.append({
        "doc_id": record["doc_id"],
        "answer": answer,
        "raw_answer": raw_answer,
        "json_as_returned": json_as_returned,
        "schema_valid": schema_valid,
        "error": error,
        "primary_ip_area": primary,
        "primary_correct": primary in record["metadata"]["ip_areas"],
        "seconds": round(seconds, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
    })
    print(f"{i + 1}/{len(records)}  {record['doc_id']}  {seconds:.1f}s  "
          f"{output_tokens} tokens  {'valid' if schema_valid else 'INVALID'}")


# save and summarise:

os.makedirs("outputs", exist_ok=True)
with open(f"outputs/{RUN_NAME}.jsonl", "w") as f:
    for r in results:
        f.write(json.dumps(r) + "\n")

n = len(results)
print("\nValid JSON exactly as returned (LLMBox's check):", sum(r["json_as_returned"] for r in results), "/", n)
print("Matches the Pydantic model:", sum(r["schema_valid"] for r in results), "/", n)
print("Primary IP area correct:", sum(r["primary_correct"] for r in results), "/", n)
print("Saved to", f"outputs/{RUN_NAME}.jsonl")
