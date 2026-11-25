"""Figure 2. Single-cell atlas of the regenerating uterus.

Panels
  A UMAP coloured by stage (5), legend labels carry the per-stage cell count
  B UMAP coloured by compartment (4)
  C UMAP coloured by cell identity (16, Harmony-integrated)
  D cell-identity marker heatmap, 16 types x 139 deck genes, gene-wise z-score
  E five per-stage UMAPs, stage cells coloured by identity over a grey backdrop
  F composition bars, mean % across replicates, each type scaled to its own max

Inputs (repo_root/data/...)
  staged/cells.parquet 698,631 cells x 9 cols
  staged/composition_by_replicate.csv per-replicate percentages
  staged/marker_zscore_deck139.csv 16 x 139, written by scripts/stage_marker_deck139.py
  raw_light/palette_final.json cell-identity palette
  raw_light/Fig2d_deck_marker_order.json authoritative 139-gene column order

Entry point: build(repo_root, out_dir) -> {panel_label: path}
"""
import json
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import render_svg

STAGES = ["CTRL", "E16_5", "PPD1", "PPD7", "PPD30"]
STAGE_LABEL = {"CTRL": "CTRL", "E16_5": "E16.5", "PPD1": "PPD1", "PPD7": "PPD7",
               "PPD30": "PPD30"}
STAGE_COLOR = {"CTRL": "#808080", "E16_5": "#cb7a29", "PPD1": "#A0522D",
               "PPD7": "#FA8072", "PPD30": "#4682B4"}
COMPARTMENTS = ["Epithelial", "Stromal", "Endothelial", "Immune"]
COMPARTMENT_COLOR = {"Epithelial": "#748190", "Stromal": "#367f80",
                     "Endothelial": "#e58677", "Immune": "#c58a4c"}
BG_GREY = "#ececec"

# Deck order of the 16 named cell types, grouped by compartment.
TYPE_ORDER = [
    "Epithelial",
    "Fibroblast", "Mesothelial", "Proliferating stromal", "Stromal (remodeling)",
    "Pericyte", "Smooth muscle", "Fibroblast (Igf1+)",
    "Vascular endothelial", "Lymphatic endothelial",
    "Macrophage", "Dendritic cell (cDC1)", "NK cell", "Macrophage/APC", "T cell",
    "Dendritic cell",
]
TYPE_SHORT = {
    "Epithelial": "Epithelial", "Fibroblast": "Fibroblast",
    "Mesothelial": "Mesothelial", "Proliferating stromal": "Prolif. stromal",
    "Stromal (remodeling)": "Stromal (remod.)", "Pericyte": "Pericyte",
    "Smooth muscle": "Sm. muscle", "Fibroblast (Igf1+)": "Fib. Igf1+",
    "Vascular endothelial": "Vasc. endo.", "Lymphatic endothelial": "Lymph. endo.",
    "Macrophage": "Macrophage", "Dendritic cell (cDC1)": "cDC1",
    "NK cell": "NK cell", "Macrophage/APC": "Mac/APC", "T cell": "T cell",
    "Dendritic cell": "DC",
}
# Compartment of each named type (matches cells.parquet 'compartment').
TYPE_COMPARTMENT = {
    "Epithelial": "Epithelial",
    "Fibroblast": "Stromal", "Mesothelial": "Stromal",
    "Proliferating stromal": "Stromal", "Stromal (remodeling)": "Stromal",
    "Pericyte": "Stromal", "Smooth muscle": "Stromal",
    "Fibroblast (Igf1+)": "Stromal",
    "Vascular endothelial": "Endothelial", "Lymphatic endothelial": "Endothelial",
    "Macrophage": "Immune", "Dendritic cell (cDC1)": "Immune", "NK cell": "Immune",
    "Macrophage/APC": "Immune", "T cell": "Immune", "Dendritic cell": "Immune",
}

# Fig2d row order. The rendered reference panel
# (sc_endom_EDITABLE_SVG/Fig1/03_celltype_marker_heatmap.svg) labels its 16 rows
# ALPHABETICALLY top-to-bottom, read from the y-coordinate of each ytick label.
# That pairing of alphabetical rows with the positional column order is what
# reproduces the published diagonal-block score (|rho| = 0.771); the
# deck/compartment TYPE_ORDER used elsewhere in this figure gives 0.128.
MARKER_ROW_ORDER = sorted(TYPE_ORDER)

QC_FAIL_REPLICATE = "CTRL_3"  # 36 cells, excluded from replicate-level statistics

# Panel a's legend spells CTRL out as "Control", following the deck.
STAGE_LEGEND_LABEL = dict(STAGE_LABEL, CTRL="Control")

# Published per-stage cell counts. stage_counts() reads the real numbers from
# cells.parquet and checks them against these, so a restaged atlas fails here
# instead of shipping a legend that disagrees with the manuscript.
STAGE_N_EXPECTED = {"CTRL": 127_161, "E16_5": 17_688, "PPD1": 194_125,
                    "PPD7": 338_578, "PPD30": 21_079}


# inputs

def _p(repo_root, *parts):
    return os.path.join(repo_root, "data", *parts)


def load_cells(repo_root):
    df = pd.read_parquet(_p(repo_root, "staged", "cells.parquet"))
    seen = set(df["celltype"].unique())
    missing = [t for t in TYPE_ORDER if t not in seen]
    if missing:
        raise ValueError(f"cells.parquet lacks expected cell type(s): {missing}")
    extra = sorted(seen - set(TYPE_ORDER))
    if extra:
        raise ValueError(f"cells.parquet has unexpected cell type(s): {extra}")
    return df


def load_palette(repo_root):
    pal = json.load(open(_p(repo_root, "raw_light", "palette_final.json")))
    missing = [t for t in TYPE_ORDER if t not in pal]
    if missing:
        raise ValueError(f"palette_final.json lacks colour(s) for: {missing}")
    return {t: pal[t] for t in TYPE_ORDER}


def read_deck_order(repo_root):
    """The 139 deck genes in rendered-panel column order.

    Read fresh from disk on every call - the order has been corrected once
    already, so nothing here may cache it. Accepts the current dict form
    (authoritative 'genes' key plus provenance/validation/warning) and the
    older bare-list form.
    """
    with open(_p(repo_root, "raw_light", "Fig2d_deck_marker_order.json")) as fh:
        obj = json.load(fh)
    order = obj["genes"] if isinstance(obj, dict) else obj
    if len(order) != 139 or len(set(order)) != 139:
        raise ValueError(f"expected 139 unique deck genes, got {len(order)}")
    return list(order)


def load_marker_matrix(repo_root):
    """16 x 139 z-score matrix, columns reordered to the rendered-panel order."""
    mz = pd.read_csv(_p(repo_root, "staged", "marker_zscore_deck139.csv"), index_col=0)
    order = read_deck_order(repo_root)
    if set(mz.columns) != set(order):
        raise ValueError("marker_zscore_deck139.csv gene SET does not match the "
                         "deck order file")
    # z-scores are per-gene (computed down columns), so reordering columns is
    # lossless - no need to recompute the matrix when only the order changes.
    return mz.reindex(index=MARKER_ROW_ORDER, columns=order)


def marker_blocks(mz):
    """Contiguous marker blocks, one per row, in row order.

    The reference panel is 'cell type x top-10 markers' with shared markers
    deduplicated (16 x 10 = 160 nominal, 139 unique). Boundaries are not stored
    anywhere and the rendered panel draws no separator lines, so they are
    recovered here: partition the columns into len(mz) contiguous blocks so that
    block k maximises the summed z-score of row k. Exact DP over 16 x 139.
    """
    M = mz.to_numpy(dtype=float)
    K, N = M.shape
    if K > N:
        raise ValueError(f"cannot cut {N} columns into {K} blocks")
    cum = np.concatenate([np.zeros((K, 1)), np.cumsum(M, axis=1)], axis=1)
    NEG = -np.inf
    dp = np.full((K + 1, N + 1), NEG)
    back = np.zeros((K + 1, N + 1), dtype=int)
    dp[0, 0] = 0.0
    for k in range(1, K + 1):
        for j in range(k, N - (K - k) + 1):
            i = np.arange(k - 1, j)
            val = dp[k - 1, i] + (cum[k - 1, j] - cum[k - 1, i])
            b = int(np.argmax(val))
            dp[k, j] = val[b]
            back[k, j] = i[b]
    bounds, j = [N], N
    for k in range(K, 0, -1):
        j = int(back[k, j])
        bounds.append(j)
    bounds = bounds[::-1]
    return [(bounds[k], bounds[k + 1], mz.index[k]) for k in range(K)]


def composition_means(repo_root):
    """Mean/SD % per (celltype, stage) across passing replicates, plus the points."""
    comp = pd.read_csv(_p(repo_root, "staged", "composition_by_replicate.csv"))
    keep = comp[(~comp["qc_fail"]) & (comp["replicate"] != QC_FAIL_REPLICATE)].copy()
    g = keep.groupby(["celltype", "stage"], observed=True)["percent"]
    stat = g.agg(mean="mean", sd="std", n="count").reset_index()
    return stat, keep


# UMAP helpers

def _umap_axis(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("UMAP 1", fontsize=6)
    ax.set_ylabel("UMAP 2", fontsize=6)


def _shuffled(df, seed=0):
    """Random draw order so no category is systematically painted over."""
    idx = np.random.default_rng(seed).permutation(len(df))
    return df.iloc[idx]


def _scatter_by(ax, df, colkey, palette, size=0.12):
    for key, color in palette.items():
        m = (df[colkey] == key).to_numpy()
        if not m.any():
            continue
        ax.scatter(df["umap_x"].to_numpy()[m], df["umap_y"].to_numpy()[m],
                   s=size, c=color, linewidths=0, rasterized=True, zorder=2)


def _legend(ax, palette, title, ncol=1, ms=4.0, loc="center left",
            bbox=(1.01, 0.5)):
    handles = [Line2D([], [], marker="o", ls="none", ms=ms, mfc=c, mec="none",
                      label=k) for k, c in palette.items()]
    leg = ax.legend(handles=handles, title=title, loc=loc, bbox_to_anchor=bbox,
                    frameon=False, ncol=ncol, handletextpad=0.35,
                    borderaxespad=0.0, labelspacing=0.32)
    leg.get_title().set_fontsize(6.5)
    return leg


def _umap_panel(df, colkey, palette, title, legend_title, stem, figsize, ncol=1):
    fig, ax = plt.subplots(figsize=figsize)
    _scatter_by(ax, _shuffled(df), colkey, palette)
    _umap_axis(ax)
    ax.set_title(title, fontsize=8)
    _legend(ax, palette, legend_title, ncol=ncol)
    fig.tight_layout()
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def stage_counts(df):
    """Cells per stage, in deck order, checked against the published counts."""
    n = df["stage"].astype(str).value_counts()
    got = {s: int(n.get(s, 0)) for s in STAGES}
    bad = {s: (got[s], STAGE_N_EXPECTED[s]) for s in STAGES
           if got[s] != STAGE_N_EXPECTED[s]}
    if bad:
        raise ValueError(f"per-stage cell counts differ from the published "
                         f"values (got, expected): {bad}")
    return got


def panel_a(df, out_dir):
    """UMAP by stage. The legend carries each stage's cell count."""
    n = stage_counts(df)
    pal = {f"{STAGE_LEGEND_LABEL[s]} ({n[s]:,})": STAGE_COLOR[s] for s in STAGES}
    # Colour by the legend label itself, so the swatch a reader matches is the
    # same key the scatter was drawn from.
    d = df.copy()
    d["stage_legend"] = d["stage"].astype(str).map(
        {s: f"{STAGE_LEGEND_LABEL[s]} ({n[s]:,})" for s in STAGES})
    return _umap_panel(d, "stage_legend", pal, "Atlas by stage",
                       "Stage (cells)",
                       os.path.join(out_dir, "Fig2A_umap_stage"), (3.0, 2.5))


def panel_b(df, out_dir):
    pal = {c: COMPARTMENT_COLOR[c] for c in COMPARTMENTS}
    return _umap_panel(df, "compartment", pal, "Atlas by compartment", "Compartment",
                       os.path.join(out_dir, "Fig2B_umap_compartment"), (3.0, 2.5))


def panel_c(df, palette, out_dir):
    return _umap_panel(df, "celltype", palette, "Atlas by cell identity",
                       "Cell identity (Harmony)",
                       os.path.join(out_dir, "Fig2C_umap_celltype"), (3.9, 2.6))


def panel_d(mz, out_dir):
    """16 x 139 marker heatmap, viridis, black separators between marker blocks."""
    fig, ax = plt.subplots(figsize=(7.4, 2.9))
    im = ax.imshow(mz.values, aspect="auto", cmap="viridis", vmin=-2, vmax=2,
                   interpolation="nearest")
    ax.set_yticks(range(len(mz.index)))
    ax.set_yticklabels(mz.index, fontsize=5.6)
    ax.set_xticks(range(mz.shape[1]))
    ax.set_xticklabels(mz.columns, rotation=90, fontsize=3.4)
    ax.tick_params(length=1.2, pad=1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    labels = marker_blocks(mz)
    for s, _e, _lab in labels[1:]:
        ax.axvline(s - 0.5, color="black", lw=0.7, zorder=3)
    for s, e, lab in labels:
        ax.text((s + e - 1) / 2, -0.85, TYPE_SHORT.get(lab, lab), ha="center",
                va="bottom", fontsize=4.6, rotation=90)
    ax.set_title("Cell-identity markers (gene-wise z-score)", fontsize=8, pad=44)
    cb = fig.colorbar(im, ax=ax, fraction=0.016, pad=0.008, ticks=[-2, -1, 0, 1, 2])
    cb.set_label("Z-score", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    fig.tight_layout()
    stem = os.path.join(out_dir, "Fig2D_marker_heatmap")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg", labels


def panel_e(df, palette, out_dir):
    """Per-stage UMAPs: stage cells by identity, all other cells light grey."""
    fig, axes = plt.subplots(1, 5, figsize=(9.0, 2.05))
    x = df["umap_x"].to_numpy(); y = df["umap_y"].to_numpy()
    stage = df["stage"].astype(str).to_numpy()
    for ax, s in zip(axes, STAGES):
        m = stage == s
        ax.scatter(x[~m], y[~m], s=0.08, c=BG_GREY, linewidths=0, rasterized=True,
                   zorder=1)
        sub = _shuffled(df[m], seed=1)
        _scatter_by(ax, sub, "celltype", palette, size=0.14)
        _umap_axis(ax)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_title(f"{STAGE_LABEL[s]}  n={int(m.sum()):,}", fontsize=7,
                     color=STAGE_COLOR[s])
    axes[0].set_xlabel("UMAP 1", fontsize=6)
    axes[0].set_ylabel("UMAP 2", fontsize=6)
    _legend(axes[-1], palette, "Cell identity (Harmony)", ncol=1, ms=3.4)
    fig.tight_layout()
    stem = os.path.join(out_dir, "Fig2E_umap_per_stage")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_f(stat, points, out_dir):
    """Composition bars per type, each type scaled to its own maximum."""
    fig, ax = plt.subplots(figsize=(8.2, 2.7))
    step, bw = 1.0, 0.155
    centres, ticks = [], []
    rng = np.random.default_rng(2)

    # Scaling by each type's own maximum MEAN puts SD whiskers and replicate
    # points above 1.0 (up to ~2.0 for the sparse types). Derive the ceiling
    # from the data so nothing is clipped.
    top = 1.0
    for ct in TYPE_ORDER:
        s = stat[stat["celltype"] == ct].set_index("stage").reindex(STAGES)
        d = s["mean"].max()
        top = max(top, float(((s["mean"] + s["sd"].fillna(0)) / d).max()),
                  float(points.loc[points["celltype"] == ct, "percent"].max() / d))
    for i, ct in enumerate(TYPE_ORDER):
        sub = stat[stat["celltype"] == ct].set_index("stage").reindex(STAGES)
        denom = sub["mean"].max()
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError(f"no positive mean percentage for {ct}")
        base = i * step
        centres.append(base)
        for j, s in enumerate(STAGES):
            xj = base + (j - 2) * bw
            mu = sub.loc[s, "mean"] / denom
            sd = (sub.loc[s, "sd"] or 0.0) / denom
            ax.bar(xj, mu, width=bw * 0.9, color=STAGE_COLOR[s], edgecolor="black",
                   lw=0.3, zorder=2)
            ax.errorbar(xj, mu, yerr=sd, fmt="none", ecolor="black", elinewidth=0.5,
                        capsize=1.1, capthick=0.5, zorder=4)
            pv = points[(points["celltype"] == ct) & (points["stage"] == s)]
            v = pv["percent"].to_numpy() / denom
            ax.plot(xj + rng.uniform(-bw * 0.22, bw * 0.22, v.size), v, ls="none",
                    marker="o", ms=1.5, mfc="white", mec="black", mew=0.3,
                    zorder=5)
        ticks.append(TYPE_SHORT[ct])
    ax.set_xticks(centres)
    ax.set_xticklabels(ticks, rotation=45, ha="right", fontsize=5.8)
    ax.set_xlim(-0.55, len(TYPE_ORDER) - 0.45)
    ax.set_ylim(0, top * 1.06)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0][: int(top / 0.5) + 1])
    ax.set_ylabel("mean %, scaled to each type's\nown maximum", fontsize=6)
    ax.axhline(1.0, color="#999999", lw=0.4, ls=(0, (3, 3)), zorder=1)

    # compartment rule above the groups
    yr = top * 1.10
    for comp in COMPARTMENTS:
        idx = [i for i, ct in enumerate(TYPE_ORDER) if TYPE_COMPARTMENT[ct] == comp]
        x0, x1 = min(idx) - 0.36, max(idx) + 0.36
        ax.plot([x0, x1], [yr, yr], lw=2.2, color=COMPARTMENT_COLOR[comp],
                solid_capstyle="butt", clip_on=False, zorder=6)
        ax.text((x0 + x1) / 2, yr + top * 0.03, comp, ha="center", va="bottom",
                fontsize=6, color=COMPARTMENT_COLOR[comp])
    _legend(ax, {STAGE_LABEL[s]: STAGE_COLOR[s] for s in STAGES}, "Stage",
            ncol=1, ms=3.6, bbox=(1.005, 0.5))
    ax.text(0.0, -0.55, f"mean +/- SD across biological replicates; "
                        f"{QC_FAIL_REPLICATE} (36 cells, QC fail) excluded",
            transform=ax.transAxes, fontsize=5.4, ha="left", va="top",
            color="#555555")
    fig.tight_layout()
    stem = os.path.join(out_dir, "Fig2F_composition_bars")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def build(repo_root: str, out_dir: str) -> dict:
    """Render all six Fig2 panels from staged tables."""
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    mpl.rcParams["savefig.dpi"] = 300  # resolution of the rasterized scatter layers

    cells = load_cells(repo_root)
    palette = load_palette(repo_root)
    mz = load_marker_matrix(repo_root)
    stat, points = composition_means(repo_root)

    paths = {}
    paths["A"] = panel_a(cells, out_dir)
    paths["B"] = panel_b(cells, out_dir)
    paths["C"] = panel_c(cells, palette, out_dir)
    paths["D"], _ = panel_d(mz, out_dir)
    paths["E"] = panel_e(cells, palette, out_dir)
    paths["F"] = panel_f(stat, points, out_dir)
    return paths
