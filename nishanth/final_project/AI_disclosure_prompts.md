# AI Disclosure: Prompts and Responses (Final Project)

Course: 95-820 Applications of NL(X) and LLM, Fall 2026
Author: Nishanth Madthila

This file records the AI assistance used for the final project. Section 1 covers the session that drafted the final report and appendix. Section 2 lists earlier assistance, with transcripts to be attached.

## 1. Final report and appendix: Claude (Cowork, Anthropic), 5 October 2026

### Prompt 1 (me)
I pasted the full Final Project assignment brief from Canvas (Parts A to E, submission requirements, rubric).

### Clarifying questions (Claude) and my answers
- *Which part should I start on?* My answer: Research report (Part A) and the appendix datasets plus code ZIP.
- *Whose work does the final project cover?* My answer: "so lets start with immigration work for now and then combine later."
- *What format should the report be in?* My answer: PDF.
- *The Colab raw outputs and any fine-tuned-model chat logs aren't in the folder. How should I handle them?* My answer: rebuild from the notebook.

### Prompt 2 (me)
I uploaded `AS02_llmbox_colab.ipynb` and `as02_outputs.zip` with the message "here are the outputs".

### What Claude did (summary of responses)
- Read my AS01 and AS02 memos, code, notebook outputs and raw per-record JSONL files.
- Re-checked the memo numbers against the raw outputs. All matched. It also found:
  - a token-count bug (input_tokens = 2 in Part B/C runs)
  - two invented enum values in C1 ("rejected", "dismissed")
  - the trivial "always cited" baseline (44% dev / 42% eval)
  - that no record in any run had both the right label and an exact citation list
- Wrote `make_splits.py`, which reproduces my dev/eval splits exactly.
- Wrote `build_appendix.py`, which assembles the required JSONL datasets and rebuilds the round-2 per-record metrics from the notebook log.
- Wrote `make_figures.py` (4 figures).
- Drafted the report text (all sections and appendices) in my voice from my memos, and rendered it to PDF.
- Verified the references with web searches.

I reviewed the draft and edited it before submission. [Nish: describe your edits here.]

## 2. Earlier assistance (attach transcripts)
- **AS01 starter code.** The scripts I built on were drafted by their author with Claude Code (see ATTRIBUTION.md).
- **AS01 human labels.** Claude produced a first pass for the 25-record sample (instructor's bootstrap-then-compare exercise). I reviewed and corrected each label. [Attach transcript.]
- **AS02 code, notebook and memo.** [Nish: describe and attach transcripts.]
- **Part D adjudication.** AI prefilled the "should block" column, and I confirmed all 30 rows by hand. [Attach transcript.]
