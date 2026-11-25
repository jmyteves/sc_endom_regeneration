"""Ligand inputs to the epithelium from the 16-type LIANA run (Fig. 5B-C, Fig. 6A-B).

Input is the LIANA output restricted to target == Epithelial, one row per
(source cell type, ligand complex, receptor complex, stage) with the
CellPhoneDB magnitude (lr_means) and p-value. Per-ligand input is the sum of
lr_means over every non-epithelial source and every receptor, keeping pairs
with cellphone_pvals < 0.05.

Panels
  ranking_generic ligands present at all five stages ranked by their mean input,
                    collagen I (Col1a1 + Col1a2 pooled) on top (Fig. 5B)
  chords_per_stage collagen I ribbons from each stromal sender into the
                    epithelial arc, one circle per stage, shared scale (Fig. 5C)
  ranking_acquired ligands absent in control ranked by their PPD30 input (Fig. 6A)
  chord_increase senders whose TNFa input to the epithelium rose from control
                    to PPD30, ribbon width = increase (Fig. 6B)
  legend the sender cell-type dot legend for the chord circles

Entry point: build(repo_root, out_dir) -> {label: path}
Reads data/raw_light/liana/liana_16types_to_epithelium.csv
"""
import os

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.path import Path

import render_svg

STAGES = ["CTRL", "E16_5", "PPD1", "PPD7", "PPD30"]
STAGE_LABEL = ["Control", "E16.5", "PPD1", "PPD7", "PPD30"]
STAGE_COLOR = {"CTRL": "#555555", "E16_5": "#cb7a29", "PPD1": "#a0522d",
               "PPD7": "#fa8072", "PPD30": "#4682b4"}
COLLAGEN = "#6a3d9a"
TNF = "#c2185b"
BAR_GREY = "#9aa3ab"

ARC = {"Epithelial": "#748190", "Stromal": "#367f80", "Endothelial": "#e58677", "Immune": "#c58a4c"}
DOT = {"Fibroblast": "#008080", "Stromal (remodeling)": "#1e90ff", "Proliferating stromal": "#483d8b",
       "Fibroblast (Igf1+)": "#4682b4", "Pericyte": "#20b2aa", "Smooth muscle": "#556b2f",
       "Mesothelial": "#db7093", "Epithelial": "#708090", "Macrophage": "#cd853f",
       "Dendritic cell": "#ff8c00", "Macrophage/APC": "#ffd700", "Dendritic cell (cDC1)": "#8b0000",
       "T cell": "#9370db", "NK cell": "#8b008b", "Vascular endothelial": "#fa8072",
       "Lymphatic endothelial": "#a0522d"}
ORDER = {"Stromal": ["Fibroblast", "Fibroblast (Igf1+)", "Stromal (remodeling)", "Proliferating stromal",
                     "Pericyte", "Smooth muscle", "Mesothelial"],
         "Endothelial": ["Vascular endothelial", "Lymphatic endothelial"],
         "Immune": ["Macrophage", "Macrophage/APC", "Dendritic cell", "Dendritic cell (cDC1)",
                    "T cell", "NK cell"],
         "Epithelial": ["Epithelial"]}
# Arc spans in degrees; the epithelial arc sits at the top, senders around the rest.
ARCS = {"Epithelial": (62, 118), "Stromal": (126, 250), "Endothelial": (258, 300), "Immune": (308, 416)}
R_ARC, W_ARC, R_DOT = 1.0, 0.09, 0.84

# The four factors tested on organoids. TNFa is drawn in Fig. 6B; the other
# three only enter the shared ribbon scale so that Fig. 6B keeps the width it
# had in the three-ligand panel it was cut from.
FACTORS = {"TNFα": ["Tnf"], "CXCL12": ["Cxcl12"], "Laminin 411": ["Lama4", "Lamb1", "Lamc1"]}
FACTOR_COLOR = {"TNFα": TNF, "CXCL12": "#f2a900", "Laminin 411": "#1b9e77"}


def load_inputs(repo_root):
    p = os.path.join(repo_root, "data", "raw_light", "liana", "liana_16types_to_epithelium.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"LIANA table missing: {p}")
    li = pd.read_csv(p)
    need = {"source", "target", "ligand_complex", "receptor_complex", "stage", "lr_means", "cellphone_pvals"}
    miss = need - set(li.columns)
    if miss:
        raise ValueError(f"LIANA table lacks column(s): {sorted(miss)}")
    if set(li["target"]) != {"Epithelial"}:
        raise ValueError("LIANA table must hold epithelial-target rows only")
    if set(li["stage"]) != set(STAGES):
        raise ValueError(f"stages found {sorted(set(li['stage']))}, expected {STAGES}")
    return li


def ligand_by_stage(li):
    """Per-ligand input to the epithelium per stage (significant pairs, non-epithelial senders)."""
    e = li[(li["source"] != "Epithelial") & (li["cellphone_pvals"] < 0.05)]
    return (e.groupby(["ligand_complex", "stage"])["lr_means"].sum()
             .unstack(fill_value=0).reindex(columns=STAGES, fill_value=0))


def collagen_senders(li):
    """Collagen I (Col1a1 + Col1a2) input per sender cell type per stage."""
    e = li[(li["source"] != "Epithelial") & (li["cellphone_pvals"] < 0.05)
           & (li["ligand_complex"].isin(["Col1a1", "Col1a2"]))]
    return (e.groupby(["source", "stage"])["lr_means"].sum()
             .unstack(fill_value=0).reindex(columns=STAGES, fill_value=0))


def factor_senders(li):
    """Input of each organoid factor per sender per stage, all pairs (no p cut).

    For a multi-gene factor (laminin 411) the per-gene sums are averaged. This is
    the table the three-ligand chord panel was drawn from.
    """
    genes = sum(FACTORS.values(), [])
    sub = li[li["ligand_complex"].isin(genes)]
    pt = sub.groupby(["stage", "ligand_complex", "source"])["lr_means"].sum().unstack("stage").fillna(0)
    sources = [t for comp in ("Stromal", "Endothelial", "Immune", "Epithelial") for t in ORDER[comp]]
    rows = []
    for fac, gs in FACTORS.items():
        for src in sources:
            vals = {s: np.mean([pt.loc[(g, src), s] if (g, src) in pt.index else 0 for g in gs])
                    for s in STAGES}
            rows.append(dict(ligand=fac, source=src, **vals))
    return pd.DataFrame(rows)


# Ranking bars with a stage-profile heat strip

def _ranking(top, highlight, label_of, xlabel, title, xlim, xticks, hi_col, stem, left=0.30, right=0.86):
    idx = list(top.index)[::-1]
    vals = top["value"].values[::-1]
    Z = top[STAGES].div(top[STAGES].max(axis=1), axis=0).values[::-1]
    cmap = plt.get_cmap("Reds")
    fig = plt.figure(figsize=(3.3, 1.75))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.6, 1, 0.06], wspace=0.08, left=left, right=right,
                          bottom=0.24, top=0.80)
    ax = fig.add_subplot(gs[0, 0])
    ax.barh(range(len(idx)), vals, color=[hi_col if i == highlight else BAR_GREY for i in idx],
            height=0.72, lw=0)
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([label_of(i) for i in idx], fontsize=6, style="italic")
    for tl, i in zip(ax.get_yticklabels(), idx):
        if i == highlight:
            tl.set_color(hi_col); tl.set_fontweight("bold"); tl.set_style("normal")
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", labelsize=6, length=2, pad=1.5, width=0.5)
    ax.set_xlabel(xlabel, fontsize=6, labelpad=2)
    ax.set_xlim(*xlim); ax.set_xticks(xticks)
    ax.set_title(title, fontsize=6, loc="left", pad=3)
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax)
    for r in range(len(idx)):
        for c in range(5):
            ax2.add_patch(mpatches.Rectangle((c - 0.5, r - 0.5), 1, 1, facecolor=cmap(Z[r, c]),
                                             edgecolor="none"))
    ax2.set_xlim(-0.5, 4.5)
    ax2.add_patch(mpatches.Rectangle((-0.5, idx.index(highlight) - 0.5), 5, 1, fill=False,
                                     edgecolor="black", lw=0.9))
    ax2.set_xticks(range(5))
    ax2.set_xticklabels(STAGE_LABEL, fontsize=5.5, rotation=60, ha="right", rotation_mode="anchor")
    for tl, s in zip(ax2.get_xticklabels(), STAGES):
        tl.set_color(STAGE_COLOR[s])
    ax2.tick_params(axis="x", length=0, pad=1.5)
    ax2.tick_params(axis="y", left=False, labelleft=False)
    for sp in ax2.spines.values():
        sp.set_visible(False)
    ax2.set_title("Stage profile of each\nligand’s epithelial input", fontsize=6, loc="left", pad=3)
    cax = fig.add_subplot(gs[0, 2])
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(0, 1))
    cb = fig.colorbar(sm, cax=cax, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=5, length=2, width=0.5, pad=1)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Input relative to the\nligand’s peak stage (1 = peak)", fontsize=5, labelpad=2)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(stem + ".svg", transparent=True)
        fig.savefig(stem + ".png", dpi=600, transparent=True)
    plt.close(fig)
    return stem + ".svg"


def ranking_generic(M, out_dir, stem="Fig5B_collagenI_top_generic_ligand"):
    """Fig. 5B: ligands present at every stage, by mean input; collagen I pooled."""
    G = M[(M > 0).all(axis=1)].copy()
    G2 = G.copy()
    G2.loc["Collagen I"] = G2.loc["Col1a1"] + G2.loc["Col1a2"]
    G2 = G2.drop(index=["Col1a1", "Col1a2"])
    G2["value"] = G2[STAGES].mean(axis=1)
    top = G2.sort_values("value", ascending=False).head(10)
    path = _ranking(top, "Collagen I", lambda i: "Collagen I\n(Col1a1 + Col1a2)" if i == "Collagen I" else i,
                    "Mean input to epithelium across stages\n(LIANA magnitude, p < 0.05 pairs)",
                    f"Ligands present at every stage\n(top 10 of {len(G)})", (0, 140), [0, 50, 100],
                    COLLAGEN, os.path.join(out_dir, stem))
    G2.drop(columns="value").round(3).to_csv(os.path.join(out_dir, "liana_generic_ligands_all_stages.csv"))
    return path, top


def ranking_acquired(M, out_dir, stem="Fig6A_TNFa_top_acquired_PPD30"):
    """Fig. 6A: ligands absent in control, by PPD30 input; Tnf on top."""
    A = M[M["CTRL"] == 0].copy()
    A["value"] = A["PPD30"]
    top = A.sort_values("value", ascending=False).head(10)
    path = _ranking(top, "Tnf", lambda i: "Tnf (TNFα)" if i == "Tnf" else i,
                    "Input to epithelium at PPD30\n(LIANA magnitude, p < 0.05 pairs)",
                    f"Ligands absent in control,\npresent at PPD30 (top 10 of {len(A)})", (0, 17.5),
                    [0, 5, 10, 15], TNF, os.path.join(out_dir, stem), left=0.21, right=0.95)
    return path, top


# Chord circles

def _pol(r, deg):
    a = np.deg2rad(deg)
    return np.array([r * np.cos(a), r * np.sin(a)])


def _ribbon(ax, p0, w0deg, r0, p1a, p1b, color, alpha=0.6):
    s1, s2 = _pol(r0, p0 - w0deg / 2), _pol(r0, p0 + w0deg / 2)
    e1, e2 = _pol(R_ARC - 0.01, p1a), _pol(R_ARC - 0.01, p1b)
    c = (0.0, 0.0)
    verts = [tuple(s1), c, tuple(e1), tuple(e2), c, tuple(s2), tuple(s1)]
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CURVE3, Path.CURVE3, Path.CLOSEPOLY]
    ax.add_patch(mpatches.PathPatch(Path(verts, codes), facecolor=color, edgecolor="none",
                                    alpha=alpha, zorder=2))


def _circle_frame(ax, fs_comp=5):
    """Compartment arcs and sender dots; returns {cell type: angle}."""
    ax.set_aspect("equal"); ax.axis("off")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.26, 1.26)
    for comp, (a, b) in ARCS.items():
        ax.add_patch(mpatches.Wedge((0, 0), R_ARC + W_ARC, a, b, width=W_ARC, facecolor=ARC[comp],
                                    edgecolor="none", zorder=3))
        mid = (a + b) / 2 % 360
        rot = mid - 90 if 0 <= mid <= 180 else mid + 90
        ax.text(*_pol(R_ARC + W_ARC + 0.09, mid), comp, ha="center", va="center", fontsize=fs_comp,
                fontweight="bold", color=ARC[comp], rotation=rot)
    pos = {}
    for comp, types in ORDER.items():
        a, b = ARCS[comp]
        for t, ang in zip(types, np.linspace(a, b, len(types) + 2)[1:-1]):
            pos[t] = ang % 360
            ax.add_patch(mpatches.Circle(_pol(R_DOT, ang), 0.055, facecolor="white", edgecolor="none", zorder=5))
            ax.add_patch(mpatches.Circle(_pol(R_DOT, ang), 0.036, facecolor=DOT[t], edgecolor="none", zorder=6))
    return pos


def _draw_ribbons(ax, weights, pos, k, color):
    """Ribbons from each sender (weights: Series indexed by cell type) into the epithelial arc."""
    d = pd.DataFrame({"v": weights[weights > 0.05]})
    d["ang"] = d.index.map(pos)
    d = d.sort_values("ang", ascending=False)
    ea, eb = ARCS["Epithelial"]
    cur = (ea + eb) / 2 - k * d["v"].sum() / 2
    for _, r in d.iterrows():
        w = k * r["v"]
        _ribbon(ax, r["ang"], max(2.0, w * 1.3), R_DOT - 0.05, cur, cur + w, color)
        cur += w


def chords_per_stage(C, out_dir, stem="Fig5C_collagenI_chords_per_stage"):
    """Fig. 5C: collagen I input per sender, one circle per stage, one ribbon scale."""
    k = 50 / C.sum().max()      # degrees per unit; the widest stage fills ~50 deg of the arc
    fig, axes = plt.subplots(1, 5, figsize=(7.25, 1.7),
                             gridspec_kw=dict(wspace=0.02, left=0.005, right=0.995, bottom=0.01, top=0.88))
    for ax, s, lab in zip(axes, STAGES, STAGE_LABEL):
        pos = _circle_frame(ax)
        _draw_ribbons(ax, C[s], pos, k, COLLAGEN)
        ax.set_title(lab, color=STAGE_COLOR[s], fontsize=7, fontweight="bold", pad=0)
    fig.text(0.005, 0.995, "Collagen I (Col1a1 + Col1a2) sent to the epithelium; ribbon width = "
             "LIANA interaction magnitude (shared scale); senders are stromal only",
             fontsize=6, ha="left", va="top")
    p = os.path.join(out_dir, stem)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(p + ".svg", transparent=True)
        fig.savefig(p + ".png", dpi=600, transparent=True)
    plt.close(fig)
    return p + ".svg"


def chord_increase(F, out_dir, ligand="TNFα", stem="Fig6B_TNFa_chord_increase_PPD30"):
    """Fig. 6B: senders whose input of `ligand` rose from control to PPD30."""
    T = F.copy()
    T["delta"] = T["PPD30"] - T["CTRL"]
    T = T[T["source"] != "Epithelial"]
    k = 50 / T.groupby("ligand")["delta"].apply(lambda s: s.abs().sum()).max()
    fig, ax = plt.subplots(figsize=(2.2, 2.2),
                           gridspec_kw=dict(left=0.01, right=0.99, bottom=0.01, top=0.90))
    pos = _circle_frame(ax, fs_comp=5.5)
    d = T[(T["ligand"] == ligand) & (T["delta"] > 0.05)].set_index("source")["delta"]
    _draw_ribbons(ax, d, pos, k, FACTOR_COLOR[ligand])
    ax.set_title(ligand, color=FACTOR_COLOR[ligand], fontsize=7, fontweight="bold", pad=0)
    p = os.path.join(out_dir, stem)
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(p + ".svg", transparent=True)
        fig.savefig(p + ".png", dpi=600, transparent=True)
    plt.close(fig)
    return p + ".svg"


def legend(out_dir, stem="Fig6B_sender_legend", fs=7):
    """One-column dot legend of the fifteen sender cell types."""
    fig = plt.figure(figsize=(1.4, 2.2))
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off")
    handles = [plt.Line2D([], [], marker="o", ls="", color=DOT[t], label=t, ms=5)
               for comp in ("Stromal", "Endothelial", "Immune") for t in ORDER[comp]]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0, 1), frameon=False, fontsize=fs,
              ncol=1, handletextpad=0.3, labelspacing=0.45, borderaxespad=0)
    p = os.path.join(out_dir, stem)
    fig.savefig(p + ".svg", transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)
    return p + ".svg"


def build(repo_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    mpl.rcParams.update({"font.family": "Arial", "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6})
    li = load_inputs(repo_root)
    M = ligand_by_stage(li)
    C = collagen_senders(li)
    F = factor_senders(li)
    out = {}
    out["5B"], top5 = ranking_generic(M, out_dir)
    out["5C"] = chords_per_stage(C, out_dir)
    out["6A"], top6 = ranking_acquired(M, out_dir)
    out["6B"] = chord_increase(F, out_dir)
    out["6B_legend"] = legend(out_dir)
    M.round(3).to_csv(os.path.join(out_dir, "liana_epithelial_ligand_inputs_by_stage.csv"))
    C.round(3).to_csv(os.path.join(out_dir, "liana_collagenI_senders_by_stage.csv"))
    out["tables"] = {"ligand_by_stage": M, "collagen_senders": C, "factor_senders": F,
                     "top_generic": top5, "top_acquired": top6}
    return out
