# AS01 — Corpus Construction and Characterization

## Group topic and individual subtopic

**Group topic:** Legal case research

**My subtopic:** AI-assisted immigration & refugee law case research —
precedent identification and citation treatment in the immigration and
protection-visa opinions of the Australian Federal Court. This is one of
three practice-area subtopics within the group's shared broad topic; my two
groupmates are each taking a different practice area (e.g. intellectual
property, criminal law, or another area of law) so the three individual
corpora can later be combined into one coherent "legal case research" group
corpus, the same way the assignment's News example combines Sports/Political/
Business into one News corpus.

The research questions this corpus is built to support: how accurately can a
model identify the precedents an immigration case relies on, how often would
it hallucinate or mis-state a citation, and does the way a case treats a
precedent (applies it, distinguishes it, follows it, ...) show up reliably
enough in the text for a model to classify it. Downstream, this corpus is
meant to support an API that answers questions like "what precedents does
this immigration case rely on, and how does it treat each one."

## What the corpus contains

220 records built from Australian Federal Court of Australia (FCA) opinions
naming the Minister for Immigration as a party (protection-visa appeals,
Refugee Review Tribunal reviews, visa cancellations, and related migration
matters), decided 2006–2009. Each record contains the full opinion text, and,
where a matching citation record exists in the source dataset, a structured
table of the precedents that opinion cites and how it treats each one.

- 137 records (62.3%) are `modality: "mixed"` — full opinion text plus a
  `table_json` array of `{citation_class, cited_case, cited_case_url,
  citation_context}` rows, one per precedent the opinion cites.
- 83 records (37.7%) are `modality: "text"` — full opinion text only, no
  citation table (the source dataset didn't have a matching citation-class
  file for these cases).
- Every record's `metadata` also carries the case name, court, year, and the
  case's own catchphrases (short editorial tags AustLII assigns to each
  opinion) — useful context even though they aren't one of the three graded
  extraction fields.
- The corpus is filtered from a larger 3,890-case dataset: 917 cases matched
  the immigration/refugee-law filter (case name contains "Minister for
  Immigration"), and 220 were randomly sampled from those (seed 42), via
  `build_corpus.py`.

The three fields this corpus is built to test structured extraction on
(defined in `taxonomy.json`):

| Field | Kind | What it is |
|---|---|---|
| `visa_category` | label (open) | The type of immigration/visa matter the case concerns (protection/refugee visa, partner/family visa, skilled/business visa, student visa, visa cancellation, citizenship, ...) |
| `citation_treatment` | label (enum) | How the opinion treats a precedent it cites — cited, referred to, applied, followed, considered, discussed, distinguished, quoted, related, affirmed, approved |
| `cited_cases` | list | Case names/citations referenced in the opinion, copied exactly as written |

## Sources

All 220 records come from a single underlying source, documented in
`sources.csv`:

- **UCI Machine Learning Repository — "Legal Case Reports" dataset**
  (Galgani, Compton & Hoffmann), itself built from opinions published by
  [AustLII](http://www.austlii.edu.au) (the Federal Court of Australia's
  public case-law database).
- **Terms:** the dataset's own `readme.txt` states research/course use only,
  no redistribution, and asks that its papers be cited if the work is
  published (Galgani & Hoffmann, AI 2010; Galgani, Compton & Hoffmann,
  COLING/ACL 2012). This corpus is built and used under those terms — it is
  not published or redistributed beyond this assignment, consistent with the
  assignment brief's own instruction not to upload corpora publicly.

## Collection and preprocessing decisions

- **Subtopic filter:** cases were kept only if their `<name>` field mentions
  "Minister for Immigration" (or the historical "Minister for Multicultural
  and Indigenous Affairs" variant), which is how this dataset identifies the
  government party in a migration-law proceeding — there's no separate
  practice-area field in the source data to filter on directly.
- **Sampling:** 220 of the 917 matching opinions were drawn with a fixed
  random seed (42), for reproducibility.
- **Parsing:** the source XML is not strictly well-formed (e.g.
  `<catchphrase "id=c0">` instead of a proper `id="c0"` attribute), so
  `build_corpus.py` parses it with regular expressions rather than an XML
  library.
- **raw_text** is the case's full text, reconstructed by joining every
  `<sentence>` element in document order.
- **table_json** is only populated when a citation-class record exists for
  that case; each citation's context paragraph is capped to 500 characters
  to keep record size reasonable.
- **retrieved_at** reflects when this corpus was built from the already-
  downloaded UCI dataset, not the original AustLII publication date (that's
  captured separately, per-record, in `metadata.year`).
- **No PII/PHI/FIN/CONF masking was applied.** These are already-published,
  public court judgments (a matter of public record) — including asylum
  seekers' names as parties, since that's how Australian court judgments are
  published — not private data collected for this project, so they don't
  fall under the assignment's sensitive-information categories.

## Known limitations

- **Single source domain.** Every record comes from one curated dataset
  (itself sourced entirely from AustLII), so `check_corpus.py` flags "only
  one source domain" — there's no cross-source variation to compare against.
- **One jurisdiction, one court, one narrow date range.** All cases are from
  the Australian Federal Court, 2006–2009; the corpus can't speak to how a
  model handles other courts, jurisdictions (e.g. U.S. immigration case law),
  or more recent legal language and citation conventions.
- **`visa_category` has no ground-truth label in the source data** — unlike
  `citation_treatment`, which is copied from the dataset's own annotation, a
  case's visa category has to be inferred from its text, so human and model
  labels for this field are both judgment calls, not verifiable against an
  external key.
- **No outcome/disposition field.** The corpus captures what a case cites and
  how it treats those citations, but not what the case itself decided (visa
  granted, appeal dismissed, etc.), so it can't support questions like "how
  often do immigration appeals succeed."
- **Filter is name-based, not substantive.** Filtering on "Minister for
  Immigration" in the case name catches the vast majority of migration
  matters but could miss an immigration case styled unusually, or loosely
  include a rare procedural case that happens to name the Minister for an
  unrelated reason.
- **License restricts reuse.** Because the source data is for research/course
  use only, this corpus cannot be published, redistributed, or reused outside
  this assignment.
