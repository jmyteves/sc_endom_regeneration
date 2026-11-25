"""Fig. S4. Gene Ontology terms of the five programs across the four compartments.

  A GO term matrix, programs x compartments x stages

Entry point: build(repo_root, out_dir) -> {panel_label: path}
"""
import program_robustness_panels as pr


def build(repo_root, out_dir):
    out = pr.build(repo_root, out_dir, letters=["e"])
    return {"A": out["e"]}
