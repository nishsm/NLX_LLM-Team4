#!/usr/bin/env python3
"""Build corpus.jsonl for the Intellectual Property subtopic.

Source: Legal Case Reports (Galgani 2010, UCI ML Repository id 239) --
Federal Court of Australia judgments 2006-2009, harvested from AustLII.

IP = Intellectual Property 

Selection
---------
A case enters the corpus if its reporter-assigned catchphrases mention
patent / trade mark / copyright / registered design / passing off /
intellectual property. Catchphrases (short topic label / summary of case) are written by the law reporter, not
inferred, so this is a high-precision filter.

One case -> several records (assignment 2.2)
--------------------------------------------
  * 2-3 TEXT records   sections of the judgment, split on headings
  * 1   TABLE record   the authorities the case cited, with treatment class
  * some TABLE records the citation-catchphrase layer (a constructed
                       aggregation that exists only in the dataset, not on
                       the AustLII page -- hence a different source_url)

The XML has malformed attributes (<catchphrase "id=c0">), so it is parsed
with regexes rather than ElementTree.

I.e. read three folders of messy XML (dataset), decide which cases count as IP, and turn each one into several tidy JSON lines.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# --------------------------------------------------------------------------
# constants
# --------------------------------------------------------------------------

# dataset 
UCI_URL = "https://archive.ics.uci.edu/dataset/239/legal+case+reports"

LICENSE_JUDGMENT = (
    "Federal Court of Australia judgment text, published by AustLII. Obtained via the "
    "Legal Case Reports dataset (Galgani 2010, UCI ML Repository id 239). Dataset readme "
    "states research use only and no redistribution; the UCI landing page states CC BY 4.0. "
    "The stricter terms are followed here: used for coursework only, not redistributed or published."
)

LICENSE_ANNOTATION = (
    "Citation-catchphrase layer constructed by Galgani (2010) from cases citing or cited by "
    "the present case; not present on the AustLII page. Legal Case Reports dataset, UCI ML "
    "Repository id 239. Research use only per the dataset readme; not redistributed."
)

# keywords related to IP (Intellectual Property) in the reporter-assigned catchphrases to filter what to include in the corpus 
IP_AREAS = {
    "patent": r"\bpatent",
    "trade_mark": r"trade\s?marks?\b",
    "copyright": r"\bcopyright",
    "registered_design": r"\bdesigns? act\b|\bregistered design",
    "passing_off": r"passing off",
}
IP_ANY = re.compile(
    r"\bpatent|trade\s?marks?\b|\bcopyright|\bdesigns? act\b|\bregistered design"
    r"|passing off|intellectual property",
    re.I,
)

# character limits for filtering and chunking
MIN_BODY, MAX_BODY = 4_000, 150_000
MIN_CHUNK, MAX_CHUNK = 1_200, 4_500

# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------

# the XML is malformed, so we parse with regexes rather than ElementTree

def clean(text: str) -> str:
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tag_block(xml: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", xml, re.S)
    return m.group(1) if m else ""


def items(xml: str, tag: str) -> list[str]:
    return [clean(t) for t in re.findall(rf"<{tag}\b[^>]*>(.*?)</{tag}>", xml, re.S)]


def parse_fulltext(path: Path) -> dict:
    xml = path.read_text(encoding="utf-8", errors="replace")
    return {
        "name": clean(tag_block(xml, "n")),
        "url": clean(tag_block(xml, "AustLII")),
        "catchphrases": [c for c in items(tag_block(xml, "catchphrases"), "catchphrase") if c],
        "sentences": [s for s in items(tag_block(xml, "sentences"), "sentence") if s],
    }


def parse_citations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    xml = path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for block in re.findall(r"<citation\b[^>]*>(.*?)</citation>", xml, re.S):
        tocase = clean(tag_block(block, "tocase"))
        if not tocase:
            continue
        rows.append({
            "cited_case": tocase,
            "treatment_class": clean(tag_block(block, "class")) or "unspecified",
            "cited_at_austlii": clean(tag_block(block, "AustLII")),
            "citing_passage": clean(tag_block(block, "text"))[:600],
        })
    return rows


def parse_summ(path: Path) -> dict:
    if not path.exists():
        return {"legislation": [], "citphrases": []}
    xml = path.read_text(encoding="utf-8", errors="replace")
    legislation = [t for t in items(tag_block(xml, "legistitles"), "title") if t]
    citphrases = []
    for m in re.finditer(r"<citphrase\b([^>]*)>(.*?)</citphrase>", tag_block(xml, "citphrases"), re.S):
        attrs, body = m.group(1), clean(m.group(2))
        if not body:
            continue
        kind = re.search(r'type=["\']?(\w+)', attrs)
        frm = re.search(r'from=["\']?([^"\'>]+)', attrs)
        citphrases.append({
            "catchphrase": body,
            "relation": kind.group(1) if kind else "unknown",
            "from_case": (frm.group(1).strip() if frm else ""),
        })
    return {"legislation": legislation, "citphrases": citphrases}


# --------------------------------------------------------------------------
# chunking a judgment into sections
# --------------------------------------------------------------------------

# each judgement is a long document, so we split them into several records
# the split is based on headings, but if no headings are found, we split into equal-sized chunks

def is_heading(sentence: str) -> bool:
    """FCA headings: short, unnumbered, no terminal punctuation, not a question.

    Only used to *label* a section. Splitting does not depend on it, because
    roughly half the judgments in this corpus carry no detectable headings.
    """
    s = sentence.strip()
    if not (2 <= len(s.split()) <= 12) or len(s) >= 80:
        return False
    if re.match(r"^\d", s) or s.endswith((".", "?", ":", ";", ",")):
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    # headings are title case or upper case, not mid-sentence prose
    upper_share = sum(c.isupper() for c in letters) / len(letters)
    return upper_share > 0.6 or s.istitle()


def chunk_judgment(sentences: list[str], n_chunks: int = 3) -> list[tuple[str, str]]:
    """Split a judgment into `n_chunks` contiguous sections at sentence boundaries.

    Equal character budgets, so every case contributes the same number of
    retrievable passages regardless of how it is formatted. A section is
    labelled with the nearest preceding heading if one is detectable.
    """
    total = sum(len(s) for s in sentences)
    if total < MIN_CHUNK * 2:
        return [("", " ".join(sentences)[:MAX_CHUNK])] if total >= MIN_CHUNK else []

    budget = min(total / n_chunks, MAX_CHUNK)
    sections: list[tuple[str, str]] = []
    buf: list[str] = []
    heading, pending = "", ""
    size = 0

    for s in sentences:
        if is_heading(s):
            pending = s
        buf.append(s)
        size += len(s)
        if size >= budget and len(sections) < n_chunks - 1:
            sections.append((heading, " ".join(buf).strip()[:MAX_CHUNK]))
            heading, pending = pending, ""
            buf, size = [], 0

    if buf:
        sections.append((heading, " ".join(buf).strip()[:MAX_CHUNK]))
    return [(h, t) for h, t in sections if len(t) >= MIN_CHUNK]


# --------------------------------------------------------------------------
# building records
# --------------------------------------------------------------------------

# turns each selected case into several records for the corpus, and writes them to a JSONL file

def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def make_record(doc_id, source_url, license_note, *, raw_text=None, table_json=None,
                metadata=None, retrieved_at=None) -> dict:
    if table_json and raw_text:
        modality = "mixed"
    elif table_json:
        modality = "table"
    else:
        modality = "text"
    return {
        "doc_id": doc_id,
        "source_url": source_url,
        "retrieved_at": retrieved_at or now_iso(),
        "modality": modality,
        "raw_text": raw_text,
        "table_json": table_json or None,
        "license_note": license_note,
        "metadata": metadata or {},
    }


def ip_areas_of(catchphrases: list[str]) -> list[str]:
    joined = " ; ".join(catchphrases).lower()
    return [area for area, pat in IP_AREAS.items() if re.search(pat, joined)]


def case_year(name: str, fallback: str) -> int | None:
    m = re.search(r"\[(\d{4})\]", name)
    return int(m.group(1)) if m else (2000 + int(fallback[:2]) if fallback[:2].isdigit() else None)


def build(root: Path, out: Path, sources_out: Path, target_cases: int, seed: int) -> None:
    fulltext, cclass, csumm = root / "fulltext", root / "citations_class", root / "citations_summ"

    # ---- select ---------------------------------------------------------
    candidates = []
    for f in sorted(fulltext.glob("*.xml")):
        raw = f.read_text(encoding="utf-8", errors="replace")
        cps_block = tag_block(raw, "catchphrases")
        if not cps_block or not IP_ANY.search(cps_block):
            continue
        case = parse_fulltext(f)
        body_len = sum(len(s) for s in case["sentences"])
        if not (MIN_BODY <= body_len <= MAX_BODY):
            continue
        areas = ip_areas_of(case["catchphrases"])
        if not areas:
            continue
        citations = parse_citations(cclass / f.name)
        if len(citations) < 3:
            continue
        candidates.append({
            "file": f.name, "case": case, "areas": areas,
            "citations": citations, "summ": parse_summ(csumm / f.name),
            "body_len": body_len,
        })

    print(f"candidates after filtering: {len(candidates)}")

    # balance across IP areas: round-robin by primary area
    by_area: dict[str, list] = {}
    for c in candidates:
        by_area.setdefault(c["areas"][0], []).append(c)
    rng = random.Random(seed)
    for v in by_area.values():
        rng.shuffle(v)

    selected, pools = [], [list(v) for v in by_area.values()]
    while len(selected) < target_cases and any(pools):
        for pool in pools:
            if pool and len(selected) < target_cases:
                selected.append(pool.pop())
    print(f"selected cases: {len(selected)}  "
          f"({dict(Counter(c['areas'][0] for c in selected))})")

    # ---- records --------------------------------------------------------
    records, retrieved = [], now_iso()
    for idx, c in enumerate(selected, 1):
        case, stem = c["case"], c["file"].replace(".xml", "")
        base_meta = {
            "group_topic": "ai_assisted_legal_case_research",
            "subtopic": "intellectual_property_law",
            "case_name": case["name"],
            "austlii_id": stem,
            "austlii_url": case["url"],
            "year": case_year(case["name"], stem),
            "court": "Federal Court of Australia",
            "jurisdiction": "Australia (Commonwealth)",
            "ip_areas": c["areas"],
            "catchphrases": case["catchphrases"][:25],
        }

        # text records: judgment sections
        for j, (heading, body) in enumerate(chunk_judgment(case["sentences"]), 1):
            records.append(make_record(
                f"ip_{idx:03d}_{stem}_s{j}", UCI_URL, LICENSE_JUDGMENT,
                raw_text=body, retrieved_at=retrieved,
                metadata={**base_meta, "document_type": "judgment_section",
                          "section_heading": heading or None, "section_index": j},
            ))

        # table record: authorities cited, with treatment class
        records.append(make_record(
            f"ip_{idx:03d}_{stem}_cit", UCI_URL, LICENSE_JUDGMENT,
            table_json=c["citations"][:40], retrieved_at=retrieved,
            metadata={**base_meta, "document_type": "citation_table",
                      "n_citations": len(c["citations"][:40]),
                      "treatment_classes": sorted({r["treatment_class"]
                                                   for r in c["citations"][:40]})},
        ))

        # table record: legislation cited (every third case, to hold the text share up)
        legislation = c["summ"]["legislation"]
        if idx % 5 == 0 and len(legislation) >= 3:
            rows = [{"legislation_title": t,
                     "is_section": bool(re.search(r"\bSECT\b", t))} for t in legislation[:40]]
            records.append(make_record(
                f"ip_{idx:03d}_{stem}_leg", UCI_URL, LICENSE_JUDGMENT,
                table_json=rows, retrieved_at=retrieved,
                metadata={**base_meta, "document_type": "legislation_table",
                          "n_titles": len(rows)},
            ))

        # table record: constructed citation-catchphrase layer (dataset, not AustLII)
        citphrases = c["summ"]["citphrases"]
        if idx % 6 == 0 and len(citphrases) >= 5:
            records.append(make_record(
                f"ip_{idx:03d}_{stem}_citph", UCI_URL, LICENSE_ANNOTATION,
                table_json=citphrases[:40], retrieved_at=retrieved,
                metadata={**base_meta, "document_type": "citation_catchphrase_table",
                          "n_citphrases": len(citphrases[:40]),
                          "relations": sorted({r["relation"] for r in citphrases[:40]})},
            ))

    # ---- write ----------------------------------------------------------
    with open(out, "w", encoding="utf-8") as handle:
        for r in records:
            handle.write(json.dumps(r, ensure_ascii=False) + "\n")

    kinds = Counter(r["modality"] for r in records)
    doctypes = Counter(r["metadata"]["document_type"] for r in records)
    tabular = sum(1 for r in records if r["modality"] in ("table", "mixed") and r["table_json"])
    print(f"\nwrote {len(records)} records to {out}")
    print(f"  modality:   {dict(kinds)}")
    print(f"  doc types:  {dict(doctypes)}")
    print(f"  tabular:    {tabular}/{len(records)} = {tabular/len(records):.1%}")

    # ---- sources.csv ----------------------------------------------------
    # One source: the dataset as downloaded. AustLII is where Galgani harvested
    # the judgments in 2012, which is provenance history, not our retrieval path
    # -- each case's AustLII link is preserved in metadata.austlii_url instead.
    with open(sources_out, "w", newline="", encoding="utf-8") as handle:
        w = csv.writer(handle)
        w.writerow(["source_url", "source_name", "retrieved_at", "license_note", "record_count"])
        w.writerow([UCI_URL,
                    "Legal Case Reports (Galgani 2010), UCI ML Repository id 239 - "
                    "Federal Court of Australia judgments 2006-2009, originally from AustLII",
                    retrieved, LICENSE_JUDGMENT, len(records)])
    print(f"  wrote {sources_out} (1 source, {len(records)} records)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, default=Path("corpus"),
                   help="the dataset's corpus/ folder")
    p.add_argument("--out", type=Path, default=Path("corpus.jsonl"))
    p.add_argument("--sources", type=Path, default=Path("sources.csv"))
    p.add_argument("--cases", type=int, default=56)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    build(a.root, a.out, a.sources, a.cases, a.seed)


if __name__ == "__main__":
    main()
