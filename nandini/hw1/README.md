# Assignment 1: Intellectual Property Law Corpus

**Group topic:** AI-assisted legal case research
**My subtopic:** Intellectual property law (Federal Court of Australia, 2006–2009)
**Nandini Chanda** · 95-820 Applications of NL(X) and LLMs

## What is in the corpus

218 records built from 50 judgments. Each case produces several records, since one
judgment holds a few different kinds of information.

| Record type | Modality | Count | What it holds |
|---|---|---|---|
| `judgment_section` | text | 150 | One third of a judgment, cut at sentence boundaries |
| `citation_table` | table | 50 | Cases the judgment cited, with treatment class and the citing passage |
| `legislation_table` | table | 10 | Acts and provisions cited |
| `citation_catchphrase_table` | table | 8 | Catchphrases from cases citing or cited by this one |

Tabular share is 31.2% (68 of 218), against the 20% minimum.

By primary IP area: copyright 14 cases, patent 14, trade mark 13, passing off 5,
registered design 4.

Every record has the eight required top-level fields. I put the case-specific
information in `metadata`: `case_name`, `austlii_url`, `year`, `court`,
`jurisdiction`, `ip_areas`, `document_type`, and the reporter's `catchphrases`.

## Where it came from

Legal Case Reports (Galgani, 2010), UCI Machine Learning Repository dataset 239.
<https://archive.ics.uci.edu/dataset/239/legal+case+reports>

3,890 Federal Court of Australia judgments from 2006 to 2009, originally scraped
from AustLII. Each case comes as three files: the judgment text with catchphrases,
a list of cited cases with citation classes, and a summary layer with legislation
titles and citation catchphrases.

On licensing: The dataset readme says research use only and no redistribution.
The UCI page says CC BY 4.0. These contradict each other, so I followed the
stricter one: coursework only, nothing redistributed or published anywhere.

I set `source_url` on every record to the UCI dataset rather than to AustLII,
because UCI is where I actually downloaded the data. AustLII is where Galgani got
it in 2012, which is history rather than my retrieval path, so each case's AustLII
link sits in `metadata.austlii_url` instead.

## How I built it

`build_corpus.py` does the collection:

```bash
python build_corpus.py --root path/to/corpus --cases 50 --seed 0
```

**Picking the cases:** I filtered on the catchphrases rather than the judgment
text. Catchphrases are short topic labels written by the law reporter, so a case
tagged "infringement of copyright" really is a copyright case, whereas searching
the body would pull in any judgment that happens to say the word. A case had to
have an IP catchphrase, a body between 4,000 and 150,000 characters, and at least
three cited cases. 219 cases match on catchphrases, 110 survive all three filters,
and I take 50 of those, drawn round-robin across IP areas so one area does not
dominate.

**Splitting the judgments:** These run to about 38,000 characters on average, which
is far too long for Phi and too long to be a sensible retrieval unit anyway. I cut
each one into three roughly equal sections at sentence boundaries, capped at 4,500
characters.

My first version tried to cut at section headings, which would have been better,
but only about half these judgments have headings a script can find, so most cases
came out as a single giant chunk. Cutting by length works on every case. The
heading detection is still in the code, but now it only labels a section when a
heading happens to sit at the start of one.

**Parsing:** The XML in this dataset is malformed — attributes are written
`<catchphrase "id=c0">` with the quote in the wrong place, which makes Python's XML
parser refuse the file outright. So the parsing is all regex. The readme also
documents the case-name tag as `<n>`, but every one of the 3,890 files actually
uses `<name>`, which cost me a rebuild when I noticed all my case names were empty.

## Decisions worth flagging

**I did not mask anything.** These are published court judgments, which are public
by design and already pseudonymised where a court ordered it. More to the point,
the party names *are* the case citations, so masking "Review Australia Pty Ltd v
Redberry Enterprise" would destroy the exact thing my group wants to measure. There
is no health or financial data in these documents.

**Sections are truncated at 4,500 characters**, so the tail of a long third is lost.
A side effect is that every text record is exactly 4,500 characters, which is why
the length distribution in `corpus_stats.py` looks so flat.

**Numbers stay as strings** in `table_json`, following the starter code's
convention.

**Citation tables are capped at 40 rows** to keep records manageable.

## The extraction task

Defined in `taxonomy.json`:

| Field | Kind | Values |
|---|---|---|
| `ip_area` | label | patent, trade_mark, copyright, registered_design, passing_off, not_stated |
| `outcome` | label | granted, refused, dismissed, allowed, not_stated |
| `cited_cases` | list | case names, copied exactly |
| `legislation_cited` | list (optional) | Acts and provisions, copied exactly |

**Rules I used when labelling** the 25 evaluation records (seed 7):

1. An IP area counts if the passage itself mentions it, even in passing. Not if
   only the catchphrases do. Phi only sees `raw_text`, so labelling from the
   catchphrases would score it against information it never got.
2. Only an outcome the passage actually states. No inferring from context.
3. Legislation as the text names it, with the year and jurisdiction suffix where
   they are given. I did not resolve bare references like "the Act".
4. On citation tables, the `cited_case` value copied exactly, citation numbers
   included.

Rule 3 turned out to be underspecified — some records have both a short and a full
form of the same Act, and I recorded both. That comes up again in the memo, since
it is the source of my worst field.

## Files

| File | |
|---|---|
| `corpus.jsonl` | The 218 records |
| `sources.csv` | Source, licence, record count |
| `human_labels.jsonl` | My 25 reference labels |
| `taxonomy.json` | Extraction fields and all three prompts |
| `extraction.py` | The graded run (course starter code) |
| `memo.pdf` | Findings |

Scripts I wrote for this assignment:

| Script | |
|---|---|
| `build_corpus.py` | Builds `corpus.jsonl` and `sources.csv` from the dataset |
| `label_helper.py` | Prints the evaluation records in full; checks my labels for blanks and typos |
| `recovery_only.py` | Diagnoses the v1 failures and reruns them under a new prompt |
| `cluster_only.py` | Runs the clustering step on its own |
| `llm_only.py` | Runs the hosted-LLM bonus on its own |

The last three exist because `extraction.py` has no resume flag and its clustering
step segfaults on my machine — scikit-learn and the HuggingFace tokenizer fight
over OpenMP under Anaconda on macOS — so anything after that step was unreachable
in one run.

## Rerunning it

```bash
python build_corpus.py --root path/to/corpus --cases 50 --seed 0
python check_corpus.py corpus.jsonl --sources sources.csv
python make_human_labels.py --n 25 --seed 7      # then label by hand
python label_helper.py --check
python extraction.py                              # Phi, evaluation, recovery
python cluster_only.py --k 10
python llm_only.py --provider groq
```

Settings: Phi-4-mini-instruct, bfloat16 on MPS, greedy decoding, `max_new_tokens`
512, and `MAX_TEXT_CHARS` lowered from 3000 to **1400**. That last change was
necessary — on MPS, anything above roughly 2,900 characters made the model emit
NaN and produce garbage. The memo covers this. Hosted comparison used
`openai/gpt-oss-120b` via Groq.

## What this corpus cannot do

- **One source, one court, one four-year window.** Nothing from state courts, the
  High Court, other countries, or the last fifteen years of Australian IP law.
- **Documents are fragments.** Records are thirds of judgments capped at 4,500
  characters, and Phi saw only the first 1,400. An argument that runs across a cut
  is split between records, and no single record holds complete reasoning.
- **Two areas are too thin to say anything about.** Registered design has 4 cases
  and passing off 5.
- **No subsequent treatment.** I record what each judgment cited, not how later
  courts treated it, so the corpus cannot tell you whether a precedent still
  stands.
- **A third of it is invisible to text analysis.** 68 records have `raw_text` set
  to null, so TF-IDF gives them all the same empty vector.
- **Ambiguous licensing**, as above.

## References

Galgani, F. (2012). *Legal Case Reports* [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5ZS41

Anthropic. (2026). *Claude Opus 5*. https://claude.ai —
help with the code. 
