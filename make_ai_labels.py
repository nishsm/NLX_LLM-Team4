#!/usr/bin/env python3
"""Generate a first-pass 'AI labels' file (Claude reading the cases), kept
separate from human_labels.jsonl so the student's own hand-labeled version
can be compared against it later, per the instructor's bootstrap-then-compare
exercise.

citation_treatment / cited_cases are derived mechanically from table_json
(the dataset's own citation annotations) where available: citation_treatment
is the most common citation_class among a case's citations, and cited_cases
is the full list of cases it cites. visa_category has no source-data ground
truth, so it's Claude's own reading of each case, hand-set below. For the
handful of records with no table_json (no citation data in the source
dataset), citation_treatment/cited_cases were set from Claude's own reading
of the opinion text and are flagged as lower-confidence.
"""
import json
from collections import Counter

VISA_CATEGORY = {
    "legal_06_1190": "protection/refugee visa",
    "legal_06_1411": "none (Freedom of Information Act request concerning an immigration department file -- not a visa application)",
    "legal_06_1495": "protection/refugee visa",
    "legal_06_1694": "bridging visa (Class WE, following a student visa cancellation)",
    "legal_06_328": "partner/spouse visa (subclass 820)",
    "legal_06_402": "protection/refugee visa",
    "legal_06_653": "protection/refugee visa",
    "legal_06_78": "protection/refugee visa",
    "legal_07_157": "protection/refugee visa",
    "legal_07_32": "protection/refugee visa",
    "legal_07_909": "student visa",
    "legal_08_1156": "skilled visa (Skilled-Independent Overseas Student, subclass 880)",
    "legal_08_1306": "protection/refugee visa",
    "legal_08_1772": "protection/refugee visa",
    "legal_08_246": "protection/refugee visa",
    "legal_08_256": "protection/refugee visa",
    "legal_08_458": "visa cancellation (character test, s501 -- held a Special Category visa subclass 444)",
    "legal_08_726": "protection/refugee visa",
    "legal_08_734": "protection/refugee visa",
    "legal_08_91": "protection/refugee visa",
    "legal_09_1187": "temporary activity visa (Cultural/Social -- Sport, subclass 421)",
    "legal_09_1278": "protection/refugee visa",
    "legal_09_1340": "protection/refugee visa",
    "legal_09_842": "protection/refugee visa",
    "legal_09_897": "protection/refugee visa",
}

# manual override for the records with no table_json (no citation data in
# the source dataset) -- Claude's own reading of the raw text, lower
# confidence than the table_json-derived records
MANUAL = {
    "legal_06_1411": {"citation_treatment": "considered", "cited_cases": [
        "Brouwer v Titan Corporation Ltd & Ors (1997) 149 ALR 50",
        "Licul v Corney (1976) 50 ALJR 439",
        "Computer Edge Pty Ltd v Apple Computer Inc [1984] HCA 47 ; (1984) 54 ALR 767",
        "Hall v the Nominal Defendant [1966] HCA 36 ; (1966) 117 CLR 423",
    ]},
    "legal_06_78": {"citation_treatment": "referred to", "cited_cases": [
        "SAAP v Minister for Immigration and Multicultural and Indigenous Affairs [2005] HCA 24 ; [2005] 79 ALJR 1009",
    ]},
    "legal_07_909": {"citation_treatment": "followed", "cited_cases": [
        "Murphy v Minister for Immigration and Multicultural and Indigenous Affairs [2004] FCA 657 ; (2004) 135 FCR 550",
        "Xie v Minister for Immigration and Multicultural and Indigenous Affairs (2005) FCAFC 172",
        "Décor Corp Pty Ltd v Dart Industries Inc (1991) 33 FCR 397",
        "Deighton v Telstra Corp Ltd [1997] FCA 1568",
    ]},
    "legal_08_734": {"citation_treatment": "cited", "cited_cases": []},
    "legal_09_1187": {"citation_treatment": "cited", "cited_cases": []},
    "legal_09_1278": {"citation_treatment": "considered", "cited_cases": [
        "SZNOR v Minister for Immigration & Anor (No 2) [2009] FMCA 726",
    ]},
    "legal_09_1340": {"citation_treatment": "followed", "cited_cases": [
        "Kuruwitage v Minister for Immigration and Citizenship [2007] FCA 795",
        "Jess v Scott (1986) 12 FCR 187",
        "Hunter Valley Development Pty Ltd v Cohen (1984) 3 FCR 355",
        "Australian Prudential Regulation Authority v Holloway (2001) 48 ATR 59",
    ]},
    "legal_09_842": {"citation_treatment": "cited", "cited_cases": []},
    "legal_09_897": {"citation_treatment": "cited", "cited_cases": [
        "Minister for Immigration and Ethnic Affairs v Guo [1997] HCA 22 ; (1997) 191 CLR 559",
    ]},
}

LOW_CONFIDENCE = set(MANUAL.keys())


def main():
    by_id = {}
    with open("corpus.jsonl") as f:
        for line in f:
            r = json.loads(line)
            by_id[r["doc_id"]] = r

    ids = [json.loads(l)["doc_id"] for l in open("human_labels.jsonl")]

    out = []
    for doc_id in ids:
        r = by_id[doc_id]
        table = r.get("table_json")
        if doc_id in MANUAL:
            citation_treatment = MANUAL[doc_id]["citation_treatment"]
            cited_cases = MANUAL[doc_id]["cited_cases"]
        else:
            classes = Counter(c["citation_class"] for c in table if c.get("citation_class"))
            citation_treatment = classes.most_common(1)[0][0] if classes else ""
            cited_cases = [c["cited_case"] for c in table]

        out.append({
            "doc_id": doc_id,
            "human_label": {
                "visa_category": VISA_CATEGORY[doc_id],
                "citation_treatment": citation_treatment,
                "cited_cases": cited_cases,
            },
            "_labeled_by": "claude (first pass, per instructor bootstrap-then-compare exercise)",
            "_low_confidence": doc_id in LOW_CONFIDENCE,
        })

    with open("human_labels_ai_claude.jsonl", "w", encoding="utf-8") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out)} AI-labeled records to human_labels_ai_claude.jsonl")
    print(f"  low-confidence (no citation data in source dataset): {sorted(LOW_CONFIDENCE)}")


if __name__ == "__main__":
    main()
