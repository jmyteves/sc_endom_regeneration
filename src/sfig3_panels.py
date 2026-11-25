"""Fig. S3. Transcriptional program robustness.

  A control-anchored volcanoes, one per postpartum stage
  B program mean z against the permutation null band
  C program overlap: shared gene counts and Jaccard index
  D program heatmaps by compartment

Entry point: build(repo_root, out_dir) -> {panel_label: path}
"""
import program_robustness_panels as pr

LETTERS = {"a": "A", "b": "B", "c": "C", "d": "D"}


def build(repo_root, out_dir):
    out = pr.build(repo_root, out_dir, letters=list(LETTERS))
    return {LETTERS[k]: v for k, v in out.items()}
