# score.py
# Scores a run file (e.g. outputs/baseline1.jsonl) against the correct answers.
#
# What it measures:
#   1. Format:      did the answer contain all three fields?
#   2. IP areas:    compared with the correct labels in the corpus (metadata "ip_areas")
#   3. Cases cited: are the citations the model lists really in the text? (made-up = hallucinated)
#                   and did it find the citations that are in the text?
#   4. Legislation: are the Acts the model lists really in the text?
#   5. Cost:        time and tokens per record

import json
import re
import statistics

# ---- change this for each run ----
RUN_FILE = "outputs/custom4_loopguard.jsonl"   # the run file to score
MAX_NEW_TOKENS = 768   # the setting used in that run
# ----------------------------------

# load the corpus so we can look up each record's text and correct labels
corpus = {}
with open("data/ip_corpus/corpus.jsonl") as f:
    for line in f:
        r = json.loads(line)
        corpus[r["doc_id"]] = r

runs = []
with open(RUN_FILE) as f:
    for line in f:
        runs.append(json.loads(line))

# patterns for case citations, e.g. "[2008] FCA 803" and "(2003) 59 IPR 191"
CITATION = r"\[\d{4}\] [A-Z][A-Za-z]* \d+|\(\d{4}\) \d+ [A-Z][A-Za-z]* \d+"
# pattern for an Act name, e.g. "Trade Marks Act 1995"
ACT = r"[A-Z][A-Za-z ]*? Act \d{4}"


def get_field(answer, name):
    """Return the text after 'NAME:' on its line, or None if the field is missing."""
    match = re.search(name + r":\s*(.*)", answer)
    return match.group(1).strip() if match else None


format_ok = 0
ip_exact = 0
ip_correct_labels = ip_predicted_labels = ip_gold_labels = 0
over_labeled = 0
cit_in_text_total = cit_found = 0
cit_listed = cit_grounded = 0
act_listed = act_grounded = 0
hit_limit = 0
scored = []

for run in runs:
    record = corpus[run["doc_id"]]
    text = record["raw_text"]
    answer = run["answer"]

    ip_field = get_field(answer, "IP_AREAS")
    cases_field = get_field(answer, "CASES_CITED")
    leg_field = get_field(answer, "LEGISLATION_CITED")
    if ip_field is not None and cases_field is not None and leg_field is not None:
        format_ok += 1

    # --- IP areas ---
    gold = set(record["metadata"]["ip_areas"])
    predicted = set()
    if ip_field:
        for label in ip_field.split(","):
            predicted.add(label.strip().lower().replace(" ", "_"))
    if predicted == gold:
        ip_exact += 1
    if predicted - gold:
        over_labeled += 1
    ip_correct_labels += len(predicted & gold)
    ip_predicted_labels += len(predicted)
    ip_gold_labels += len(gold)

    # --- case citations ---
    in_text = set(re.findall(CITATION, text))
    listed = set(re.findall(CITATION, cases_field or ""))
    cit_in_text_total += len(in_text)
    cit_found += len(in_text & listed)
    cit_listed += len(listed)
    made_up = listed - in_text
    cit_grounded += len(listed) - len(made_up)

    # --- legislation ---
    acts = re.findall(ACT, leg_field or "")
    acts_made_up = [a for a in acts if a.lower() not in text.lower()]
    act_listed += len(acts)
    act_grounded += len(acts) - len(acts_made_up)

    if run["output_tokens"] >= MAX_NEW_TOKENS - 2:
        hit_limit += 1

    scored.append({
        "doc_id": run["doc_id"],
        "gold_ip": sorted(gold),
        "predicted_ip": sorted(predicted),
        "ip_exact": predicted == gold,
        "citations_in_text": sorted(in_text),
        "citations_listed": sorted(listed),
        "citations_made_up": sorted(made_up),
        "acts_made_up": acts_made_up,
    })


def pct(a, b):
    return f"{100 * a / b:.1f}% ({a}/{b})" if b else "n/a (0/0)"


n = len(runs)
print(f"Run: {RUN_FILE}   ({n} records)\n")
print("FORMAT")
print("  all three fields present:", pct(format_ok, n))
print("  hit the token limit (cut off):", pct(hit_limit, n))
print("\nIP AREAS")
print("  exact match:", pct(ip_exact, n))
print("  precision (labels given that were right):", pct(ip_correct_labels, ip_predicted_labels))
print("  recall (right labels that were given):", pct(ip_correct_labels, ip_gold_labels))
print("  records with at least one wrong extra label:", pct(over_labeled, n))
print("\nCASES CITED")
print("  citations listed that are really in the text:", pct(cit_grounded, cit_listed))
print("  made-up citations:", cit_listed - cit_grounded)
print("  citations in the text that the model found:", pct(cit_found, cit_in_text_total))
print("\nLEGISLATION")
print("  Acts listed that are really in the text:", pct(act_grounded, act_listed))
print("\nCOST")
seconds = [r["seconds"] for r in runs]
print("  median seconds per record:", round(statistics.median(seconds), 1))
print("  mean seconds per record:", round(statistics.mean(seconds), 1))
print("  mean input tokens:", round(statistics.mean(r["input_tokens"] for r in runs)))
print("  mean output tokens:", round(statistics.mean(r["output_tokens"] for r in runs)))

out_file = RUN_FILE.replace(".jsonl", "_scored.jsonl")
with open(out_file, "w") as f:
    for s in scored:
        f.write(json.dumps(s) + "\n")
print("\nPer-record details saved to", out_file)
