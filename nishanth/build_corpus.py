#!/usr/bin/env python3
"""Build corpus.jsonl and sources.csv from the UCI Legal Case Reports dataset.

Source: Galgani, Compton & Hoffmann -- Australian Federal Court opinions
(2006-2009) from AustLII, with per-case catchphrases (fulltext/) and, for
most cases, a labelled citation list (citations_class/) giving the
treatment ("cited", "applied", "followed", ...) of each precedent the case
cites.

Run from the folder that contains fulltext/ and citations_class/:

    python3 build_corpus.py --data-dir "." --out corpus.jsonl --sources sources.csv

The XML in this dataset is not strictly well-formed (attributes are written
like `<catchphrase "id=c0">` rather than `<catchphrase id="c0">`), so this
parses with regular expressions instead of an XML library.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path

GROUP_TOPIC = "legal case research"
SUBTOPIC = "AI-assisted immigration & refugee law case research (Australian Federal Court opinions, precedent identification and citation treatment)"

# Cases naming the immigration minister as a party -- this is how migration/
# protection-visa appeals are identified in this dataset (there is no
# separate practice-area field in the source data).
NAME_FILTER = re.compile(
    r"Minister for Immigration|Minister for Multicultural and Indigenous Affairs",
    re.I,
)
LICENSE_NOTE = (
    "UCI Machine Learning Repository, 'Legal Case Reports' dataset "
    "(Galgani, Compton & Hoffmann; cases sourced from AustLII, the Federal "
    "Court of Australia's public opinions). Per the dataset's own README: "
    "for research/course use only, do not redistribute; cite Galgani & "
    "Hoffmann (AI 2010) or Galgani, Compton & Hoffmann (COLING/ACL 2012) "
    "if published."
)
COURT = "Federal Court of Australia"

#  the leading "<" plus a boundary right after the tag name (whitespace or
#  ">") stops this from matching inside a plural container tag -- "sentence"
#  is a prefix of "sentences", "catchphrase" of "catchphrases", "citation"
#  of "citations", so without that boundary check this silently swallows
#  the container tag and corrupts the first match in every document.
TAG = r"<{tag}(?:\s[^>]*)?>(.*?)</{tag}>"


def _findall(tag: str, text: str) -> list[str]:
    return re.findall(TAG.format(tag=re.escape(tag)), text, re.S)


def _find(tag: str, text: str) -> str | None:
    m = re.search(TAG.format(tag=re.escape(tag)), text, re.S)
    return _clean(m.group(1)) if m else None


def _clean(s: str) -> str:
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_fulltext(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    name = _find("name", text)
    if not name:
        return None
    austlii = _find("AustLII", text)
    catchphrases = [_clean(c) for c in _findall("catchphrase", text)]
    sentences = [_clean(s) for s in _findall("sentence", text)]
    raw_text = " ".join(sentences)
    return {"name": name, "source_url": austlii, "catchphrases": catchphrases, "raw_text": raw_text}


def parse_citations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for block in re.findall(r"<citation[^>]*>(.*?)</citation>", text, re.S):
        cls = _find("class", block)
        tocase = _find("tocase", block)
        url = _find("AustLII", block)
        ctx = _find("text", block)
        if not tocase:
            continue
        row = {"citation_class": cls or "", "cited_case": tocase}
        if url:
            row["cited_case_url"] = url
        if ctx:
            row["citation_context"] = ctx[:500]
        rows.append(row)
    return rows


def year_from_stem(stem: str) -> int | None:
    m = re.match(r"(\d{2})_", stem)
    if not m:
        return None
    yy = int(m.group(1))
    return 2000 + yy if 0 <= yy <= 30 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=".", help="folder containing fulltext/ and citations_class/")
    parser.add_argument("--out", default="corpus.jsonl")
    parser.add_argument("--sources", default="sources.csv")
    parser.add_argument("--n", type=int, default=220, help="target corpus size (150-300 required)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    fulltext_dir = data_dir / "fulltext"
    citations_dir = data_dir / "citations_class"
    if not fulltext_dir.exists():
        raise SystemExit(f"{fulltext_dir} not found -- run this from the folder with fulltext/ in it")

    all_stems = sorted(p.stem for p in fulltext_dir.glob("*.xml"))

    # Filter to the immigration & refugee law subtopic before sampling, by
    # checking each case's <name> against NAME_FILTER. This is a cheap
    # pre-pass (name only, not the full parse) so we don't waste time
    # fully parsing cases outside the subtopic.
    stems = []
    for stem in all_stems:
        text = (fulltext_dir / f"{stem}.xml").read_text(encoding="utf-8", errors="replace")
        name = _find("name", text)
        if name and NAME_FILTER.search(name):
            stems.append(stem)
    print(f"{len(stems)} of {len(all_stems)} cases match the immigration & refugee law subtopic")

    rng = random.Random(args.seed)
    rng.shuffle(stems)

    retrieved_at = datetime.now(timezone.utc).isoformat()
    records: list[dict] = []
    n_mixed = 0

    for stem in stems:
        if len(records) >= args.n:
            break
        parsed = parse_fulltext(fulltext_dir / f"{stem}.xml")
        if not parsed or len(parsed["raw_text"]) < 50:
            continue

        citations = parse_citations(citations_dir / f"{stem}.xml")
        modality = "mixed" if citations else "text"
        if modality == "mixed":
            n_mixed += 1

        year = year_from_stem(stem)
        record = {
            "doc_id": f"legal_{stem}",
            "source_url": parsed["source_url"] or "http://www.austlii.edu.au/",
            "retrieved_at": retrieved_at,
            "modality": modality,
            "raw_text": parsed["raw_text"],
            "table_json": citations if modality == "mixed" else None,
            "license_note": LICENSE_NOTE,
            "metadata": {
                "group_topic": GROUP_TOPIC,
                "subtopic": SUBTOPIC,
                "case_name": parsed["name"],
                "court": COURT,
                "year": year,
                "catchphrases": parsed["catchphrases"],
                "n_citations": len(citations),
            },
        }
        records.append(record)

    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    tabular_share = n_mixed / len(records) if records else 0
    print(f"Wrote {len(records)} records to {args.out}")
    print(f"  mixed (tabular) modality: {n_mixed} ({tabular_share:.1%}) -- needs >= 20%")
    print(f"  text-only modality:       {len(records) - n_mixed}")

    with open(args.sources, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_url", "source_name", "retrieved_at", "license_note", "record_count"])
        writer.writerow([
            "http://www.austlii.edu.au/",
            "AustLII / Federal Court of Australia, via UCI ML Repository 'Legal Case Reports' dataset",
            retrieved_at,
            LICENSE_NOTE,
            len(records),
        ])
    print(f"Wrote {args.sources}")


if __name__ == "__main__":
    main()
