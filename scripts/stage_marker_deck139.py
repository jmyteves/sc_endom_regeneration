"""Derive data/staged/marker_zscore_deck139.csv - the Fig2d marker matrix.

Why this exists
  The original staging wrote marker_zscore.csv with the 66-gene group_map_17
  marker_signature set. The v3 deck's Fig2d draws 139 genes, whose authoritative
  column order lives in data/raw_light/Fig2d_deck_marker_order.json under the
  "genes" key. That order is read POSITIONALLY from the rendered panel
  (x-coordinate of each rotate(-90) <text> element in
  sc_endom_EDITABLE_SVG/Fig1/03_celltype_marker_heatmap.svg, sorted ascending).
  PDF text extraction returns reading order, not visual column order, and must
  not be used for this. This script regenerates the matrix for exactly those
  genes, in that order, WITHOUT touching the 14 GB h5ad: the replicate x
  celltype pseudobulk sums already carry everything needed.

Derivation
  raw_light/pseudobulk_rep_by_type.npz sums (228 x 17987) over 'REPLICATE|celltype'
    -> drop every level of CTRL_3 (36 cells, QC fail)
    -> sum counts within each of the 16 cell types (16 x 17987 raw counts)
    -> CPM per cell type, then log1p (library-size normalised)
    -> subset to the 139 deck genes in deck order
    -> z-score each gene ACROSS the 16 cell types (ddof=0)

Run: python scripts/stage_marker_deck139.py [repo_root]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

TYPE_ORDER = [
    "Epithelial",
    "Fibroblast", "Mesothelial", "Proliferating stromal", "Stromal (remodeling)",
    "Pericyte", "Smooth muscle", "Fibroblast (Igf1+)",
    "Vascular endothelial", "Lymphatic endothelial",
    "Macrophage", "Dendritic cell (cDC1)", "NK cell", "Macrophage/APC", "T cell",
    "Dendritic cell",
]
QC_FAIL_REPLICATE = "CTRL_3"

# Canonical markers that must be maximal in their own cell type's row.
ROW_MAX_CHECKS = {
    "Epithelial": ["Epcam", "Cd24a"],
    "T cell": ["Cd3g"],
    "Smooth muscle": ["Myh11", "Acta2"],
    "Vascular endothelial": ["Pecam1", "Flt1"],
    "NK cell": ["Nkg7", "Ncr1"],
}


def read_deck_order(repo_root):
    """The 139 deck genes in rendered-panel column order (never cached)."""
    p = os.path.join(repo_root, "data", "raw_light", "Fig2d_deck_marker_order.json")
    with open(p) as fh:
        obj = json.load(fh)
    order = obj["genes"] if isinstance(obj, dict) else obj
    if len(order) != 139 or len(set(order)) != 139:
        raise ValueError(f"expected 139 unique deck genes, got {len(order)} "
                         f"({len(set(order))} unique)")
    return list(order)


def diagonal_rho(mz):
    """|Spearman(column position, argmax-row index)| - the diagonal-block score."""
    from scipy.stats import spearmanr
    rows = list(mz.index)
    argmax_row = [rows.index(mz[g].idxmax()) for g in mz.columns]
    return abs(spearmanr(np.arange(mz.shape[1]), argmax_row).statistic)


def build_matrix(repo_root):
    npz = np.load(os.path.join(repo_root, "data", "raw_light",
                               "pseudobulk_rep_by_type.npz"), allow_pickle=True)
    levels = npz["levels"].astype(str)
    genes = npz["genes"].astype(str)
    sums = npz["sums"]

    reps = np.array([lv.split("|")[0] for lv in levels])
    ctypes = np.array([lv.split("|")[1] for lv in levels])
    keep = reps != QC_FAIL_REPLICATE
    print(f"levels {len(levels)}; dropped {int((~keep).sum())} "
          f"{QC_FAIL_REPLICATE} levels; kept {int(keep.sum())}")

    sums, ctypes = sums[keep], ctypes[keep]
    seen = set(ctypes)
    if seen != set(TYPE_ORDER):
        raise ValueError(f"cell-type mismatch: extra={sorted(seen - set(TYPE_ORDER))} "
                         f"missing={sorted(set(TYPE_ORDER) - seen)}")

    counts = np.vstack([sums[ctypes == ct].sum(axis=0) for ct in TYPE_ORDER])
    cpm = counts / counts.sum(axis=1, keepdims=True) * 1e6
    logcpm = np.log1p(cpm)

    order = read_deck_order(repo_root)
    gi = {g: i for i, g in enumerate(genes)}
    missing = [g for g in order if g not in gi]
    if missing:
        raise ValueError(f"deck genes absent from pseudobulk var index: {missing}")

    sub = logcpm[:, [gi[g] for g in order]]
    z = (sub - sub.mean(axis=0)) / sub.std(axis=0, ddof=0)
    return pd.DataFrame(z, index=TYPE_ORDER, columns=order)


def check_row_maxima(mz):
    lines, ok = [], True
    for ct, gs in ROW_MAX_CHECKS.items():
        for g in gs:
            top = mz[g].idxmax()
            good = top == ct
            ok &= good
            lines.append(f"{g:8s} argmax={top:22s} z={mz.loc[top, g]:+.2f}  "
                         f"({ct} z={mz.loc[ct, g]:+.2f})  "
                         f"{'PASS' if good else 'FAIL'}")
    return ok, lines


def main(repo_root="."):
    mz = build_matrix(repo_root)
    out = os.path.join(repo_root, "data", "staged", "marker_zscore_deck139.csv")
    mz.to_csv(out)
    print(f"wrote {out}  shape={mz.shape}")
    print(f"diagonal-block |rho| for this column order = {diagonal_rho(mz):.3f}")
    ok, lines = check_row_maxima(mz)
    print("\n".join(lines))
    if not ok:
        raise SystemExit("row-maxima sanity check FAILED")
    print("row-maxima sanity check: all PASS")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
