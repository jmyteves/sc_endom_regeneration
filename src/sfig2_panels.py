"""Fig. S2. Annotation validity, batch integration and clustering robustness.

  A cells passing QC per library (CTRL-3 excluded)
  B count-feature relation on library medians with the recorded fit
  C cells per stage, 14 libraries
  D per-stage QC medians: detected genes, total UMIs, mitochondrial percent
  E Harmony UMAP by cell identity and by stage; the raw-PCA half is not
     reproducible (the un-integrated UMAP was never staged) and is a placeholder
  F compartment-level concordance against the MCA uterus reference: placeholder,
     the Pearson matrix survives only as a raster in the delivered figure
  G cells recovered per replicate, stacked by cell type
  H within-replicate composition

Entry point: build(repo_root, out_dir) -> {panel_label: path or reason}
"""
import os

import matplotlib.pyplot as plt
import numpy as np

import render_svg
import atlas_validity_panels as av
import library_qc_panels as lq

STAGE_N = {"CTRL": 2, "E16_5": 3, "PPD1": 3, "PPD7": 3, "PPD30": 3}

NOT_COMPUTABLE = {
    "E_rawPCA": av.NOT_COMPUTABLE["d_rawPCA"],
    "F": av.NOT_COMPUTABLE["c"],
}


def panel_c(qc_cells, out_dir):
    """Cells per stage over the 14 analysed libraries."""
    per_stage, total, n_lib = lq.stage_totals(qc_cells)
    fig, ax = plt.subplots(figsize=(2.6, 2.3))
    x = np.arange(len(lq.STAGES))
    vals = [per_stage[s] for s in lq.STAGES]
    ax.bar(x, vals, color=[lq.STAGE_COLOR[s] for s in lq.STAGES], width=0.7, lw=0)
    for i, v in enumerate(vals):
        ax.text(i, v + total * 0.01, f"{v / 1000:.0f}k", ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lq.STAGE_LABEL[s]} (n={STAGE_N[s]})" for s in lq.STAGES],
                       rotation=45, ha="right", fontsize=6)
    ax.set_ylabel("Cells passing QC")
    ax.set_title("Per stage", fontsize=8.5)
    ax.text(0.02, 0.97, f"{total:,} cells\n{n_lib} libraries", transform=ax.transAxes,
            ha="left", va="top", fontsize=6)
    fig.tight_layout()
    stem = os.path.join(out_dir, "SFig2C_cells_per_stage")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def build(repo_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    q = lq.load_inputs(repo_root)
    out = {
        "A": lq.panel_i(q["qc_cells"], out_dir),
        "B": lq.panel_k(q["count_feature"], out_dir),
        "C": panel_c(q["qc_cells"], out_dir),
        "D": lq.panel_j(q["qc_dist"], out_dir),
    }
    a = av.load_inputs(repo_root)
    out["E"] = av.panel_d(a["cells"], a["palette"], out_dir)
    for lab, reason in NOT_COMPUTABLE.items():
        av._slug(lab, reason, out_dir)
        out[lab] = f"NOT COMPUTABLE: {reason}"
    out["G"] = av.panel_f(a["counts"], out_dir)
    out["H"] = av.panel_g(a["pct"], a["palette"], out_dir)
    return out
