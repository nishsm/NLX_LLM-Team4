#!/usr/bin/env python3
"""Rerun v2_recovery on the full 25-record evaluation sample (not just the
records that failed under v1), so we have an apples-to-apples "final
pipeline" evaluation to break down by cluster.

    python rerun_v2_clustering.py

Writes out/evaluation_v2_full.json and out/clusters_v2.json, and prints the
per-cluster accuracy table for the memo.
"""
import json
from pathlib import Path

from extraction import (
    apply_taxonomy, PROMPTS, run_phi, run_phi_evaluation, print_evaluation,
    cluster_corpus, accuracy_by_cluster, validate_corpus, load_human_labels,
)
from local_model import PhiClient

OUT = Path("out")
OUT.mkdir(exist_ok=True)

apply_taxonomy()

corpus = validate_corpus("corpus.jsonl")
human_labels = load_human_labels("human_labels.jsonl")
evaluation_set = [r for r in corpus.records if r["doc_id"] in human_labels]
print(f"Evaluation sample: {len(evaluation_set)} labelled records")

phi = PhiClient(device="auto")
print(f"\nRunning Phi on {len(evaluation_set)} records (prompt v2_recovery, full sample) ...")
rows = run_phi(evaluation_set, PROMPTS["v2_recovery"], phi)

evaluation_v2 = run_phi_evaluation(evaluation_set, human_labels, rows=rows,
                                   system_prompt=PROMPTS["v2_recovery"])
print_evaluation("HUMAN vs. PHI (v2_recovery, full 25-record sample)", evaluation_v2)
(OUT / "evaluation_v2_full.json").write_text(
    json.dumps(evaluation_v2, indent=2, ensure_ascii=False), encoding="utf-8"
)
print("  wrote out/evaluation_v2_full.json")

print(f"\n{'=' * 64}\nCLUSTERS (re-scored against v2_recovery)\n{'=' * 64}")
result = cluster_corpus(corpus.records, k=6, plot_path=str(OUT / "clusters_v2.png"))
per_cluster = accuracy_by_cluster(result, evaluation_v2)
print("\n  Phi accuracy (v2_recovery) on the labelled records, by cluster:")
for cluster_id, info in per_cluster.items():
    print(f"    cluster {cluster_id:>2}  n={info['n']:>3}  accuracy {info['accuracy']}")

(OUT / "clusters_v2.json").write_text(json.dumps({
    "k": 6,
    "silhouette": result.silhouette,
    "sizes": result.sizes(),
    "top_terms": result.top_terms,
    "phi_v2_accuracy_by_cluster": per_cluster,
}, indent=2, ensure_ascii=False), encoding="utf-8")
print("  wrote out/clusters_v2.json")
