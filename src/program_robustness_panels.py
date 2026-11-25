"""Transcriptional program robustness (fig. S3 A-D) and the program GO matrix (fig. S4 A).

    build(repo_root, out_dir, letters=None) -> {letter: svg_path}

  a  four control-anchored volcanoes, one per stage contrast, each annotated with its
     control-high and stage-high counts. No delivered table carries per-gene volcano
     coordinates, so the clouds are recomputed from the whole-tissue per-sample
     pseudobulk and the four count pairs are checked against
     MD_a_volcanoes_ctrl_anchored_source.csv before drawing.
  b  program mean gene z against a gene-label-permutation null, one axes per program.
     The grey ribbon is the null 2.5-97.5 percentile band at that stage; 24 of the 25
     program x stage cells sit outside it.
  c  program overlap: shared-gene counts and Jaccard index as a percentage (three
     decimals collide at this type size, and two decimals would print the true zero of
     Homeostasis vs Early repair the same as 0.001).
  d  the five programs re-z-scored inside each compartment, one heatmap per compartment,
     with each compartment's own gene counts on the block bar. Row order inside a block
     is per compartment, so rows do not correspond across the four panels.
  e  the 30 GO-BP terms of Fig. 3C against all 20 compartment modules (4 compartments x
     M1-M5), rows grouped by the program that contributed the term. Most cells carry no
     dot: a whole-tissue program's terms largely do not re-enrich inside a single
     compartment's temporal module.                                      (fig. S4 A)

Which table feeds d and e: the two candidate MD_ tables were rejected on their numbers.
MD_d_module_pseudobulk_heatmap_source.csv is whole-tissue with per-program row counts
553/217/977/742/786, not the figure's 668/169/736/579/783, and MD_e_program_GO_source.csv
resolves three compartments and 11 modules, not four and 20. Panels d and e therefore read
S6_programs_<compartment>_source and S6_GO_matrix_4compartments_source.

Program keys: the internal key for the fifth program stays "Remodelling" and disp()
renders it as "New Homeostasis", as in the other modules.
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy import stats

import render_svg
from compress_svg import compress_svg

STAGES = ["CTRL", "E16_5", "PPD1", "PPD7", "PPD30"]
STAGE_LAB = {"CTRL": "Control", "E16_5": "E16.5", "PPD1": "PPD1", "PPD7": "PPD7",
             "PPD30": "PPD30"}
PROGRAMS = ["Homeostasis", "Pregnancy", "Early repair", "Late repair", "Remodelling"]
PROG_PEAK = {"Homeostasis": "CTRL", "Pregnancy": "E16_5", "Early repair": "PPD1",
             "Late repair": "PPD7", "Remodelling": "PPD30"}
MOD_COL = {"Homeostasis": "#808080", "Pregnancy": "#cb7a29", "Early repair": "#A0522D",
           "Late repair": "#FA8072", "Remodelling": "#4682B4"}
COMPS = ["Epithelial", "Stromal", "Immune", "Endothelial"]
COMP_COL = {"Epithelial": "#748190", "Stromal": "#367f80",
            "Endothelial": "#e58677", "Immune": "#c58a4c"}

DISPLAY = {"Remodelling": "New Homeostasis"}
_INVERSE = {v: k for k, v in DISPLAY.items()}

GREY_NS = "#d4d4d4"
GREY_SIG = "#6f6f6f"
GREY_NULL = "#b9b9b9"
INK = "#333333"

# The four contrasts, keyed as MD_a writes them.
CONTRASTS = [("E16_5", "CTRL_vs_E16.5"), ("PPD1", "CTRL_vs_PPD1"),
             ("PPD7", "CTRL_vs_PPD7"), ("PPD30", "CTRL_vs_PPD30")]

Q_CUT, FC_CUT = 0.05, 1.0

# Deck values. Asserted, never used to pick or reorder anything: if a staged table
# disagrees the build stops rather than drawing a figure that no longer matches the deck.
DECK_VOLCANO = {"E16_5": (1361, 222), "PPD1": (2045, 985),
                "PPD7": (1368, 745), "PPD30": (1268, 1089)}
DECK_WHOLE_SIZES = {"Homeostasis": 678, "Pregnancy": 217, "Early repair": 985,
                    "Late repair": 745, "Remodelling": 789}
DECK_COMP_SIZES = {"Homeostasis": 668, "Pregnancy": 169, "Early repair": 736,
                   "Late repair": 579, "Remodelling": 783}
DECK_TOP_OVERLAP = ("Early repair", "Late repair", 189, 0.1226)

_TABLES = {
    "volc_counts": "MD_a_volcanoes_ctrl_anchored_source.csv",
    "null": "S3a_program_null_distributions_source.csv",
    "overlap": "S3d_program_overlap_source.csv",
    "membership": "S3d_program_membership_summary.csv",
    "go": "S6_GO_matrix_4compartments_source.csv",
    "go_rows": "SHARED_GO_fig4_canonical_30terms.csv",
}
_PSEUDOBULK = "Fig7_wholeTissue_persample_pseudobulk_counts.csv"


def disp(program):
    """The label a program is drawn with. Identity for the four unrenamed programs."""
    return DISPLAY.get(program, program)


def _key(label):
    """The internal key for a display label, for tables that already ship the new name."""
    return _INVERSE.get(label, label)


def _read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fig S3 input missing: {path}. Stage it from "
            f"sc_endom_EDITABLE_SVG/SFigs_latest/tables/{os.path.basename(path)}")
    return pd.read_csv(path)


def load_inputs(repo_root):
    """Read and check every staged table. Raises before any drawing on a disagreement."""
    d = os.path.join(repo_root, "data", "raw_light", "sfig3")
    out = {k: _read(os.path.join(d, fn)) for k, fn in _TABLES.items()}
    for c in COMPS:
        out["prog_" + c] = _read(os.path.join(d, f"S6_programs_{c.lower()}_source.csv"))
    out["counts"] = pd.read_csv(
        os.path.join(repo_root, "data", "raw_light", _PSEUDOBULK), index_col=0)

    v = out["volc_counts"].set_index("contrast")
    if len(v) != 4:
        raise ValueError(f"MD_a: expected 4 contrasts, got {len(v)}")
    for stage, row in CONTRASTS:
        got = (int(v.loc[row, "n_down_control_high"]), int(v.loc[row, "n_up_stage_high"]))
        if got != DECK_VOLCANO[stage]:
            raise ValueError(f"MD_a {row}: control-high/stage-high {got} disagrees with "
                             f"the deck {DECK_VOLCANO[stage]}")

    n = out["null"]
    if len(n) != 25:
        raise ValueError(f"S3a: expected 25 program x stage rows, got {len(n)}")
    n["program"] = n["program"].map(_key)
    if sorted(n.program.unique()) != sorted(PROGRAMS):
        raise ValueError(f"S3a: unexpected programs {sorted(n.program.unique())}")
    sizes = n.groupby("program").n_genes_program.first().to_dict()
    if sizes != DECK_WHOLE_SIZES:
        raise ValueError(f"S3a program sizes {sizes} disagree with the deck "
                         f"{DECK_WHOLE_SIZES}")

    ov = out["overlap"]
    ov["program_a"] = ov["program_a"].map(_key)
    ov["program_b"] = ov["program_b"].map(_key)
    diag = {r.program_a: int(r.intersection)
            for r in ov.itertuples() if r.program_a == r.program_b}
    if diag != DECK_WHOLE_SIZES:
        raise ValueError(f"S3d diagonal {diag} disagrees with the deck {DECK_WHOLE_SIZES}")
    off = ov[ov.program_a != ov.program_b].sort_values("intersection", ascending=False)
    top = off.iloc[0]
    got = (top.program_a, top.program_b, int(top.intersection), round(float(top.jaccard), 4))
    a, b, k, j = DECK_TOP_OVERLAP
    if {got[0], got[1]} != {a, b} or got[2] != k or got[3] != j:
        raise ValueError(f"S3d largest off-diagonal {got} disagrees with the deck "
                         f"{DECK_TOP_OVERLAP}")

    for c in COMPS:
        p = out["prog_" + c]
        p["program"] = p["program"].map(_key)
        if len(p) != 2935:
            raise ValueError(f"S6_programs_{c.lower()}: expected 2935 genes, got {len(p)}")
        got = p.program.value_counts().to_dict()
        if got != DECK_COMP_SIZES:
            raise ValueError(f"S6_programs_{c.lower()} program sizes {got} disagree with "
                             f"the deck {DECK_COMP_SIZES}")
        order = list(p.sort_values("row_index").program.drop_duplicates())
        if order != PROGRAMS:
            raise ValueError(f"S6_programs_{c.lower()} row blocks run {order}, "
                             f"expected {PROGRAMS}")

    g = out["go"]
    if sorted(g.compartment.unique()) != sorted(COMPS):
        raise ValueError(f"S6_GO_matrix: compartments {sorted(g.compartment.unique())}, "
                         f"expected {sorted(COMPS)}")
    if g.Term.nunique() != 30 or len(g) != 600:
        raise ValueError(f"S6_GO_matrix: {g.Term.nunique()} terms x {len(g)} rows, "
                         "expected 30 x 600")
    r = out["go_rows"]
    if len(r) != 30:
        raise ValueError(f"SHARED_GO_fig4_canonical_30terms: {len(r)} rows, expected 30")
    r["program"] = r["wave"].map(_key)
    if set(r.Term) != set(g.Term):
        raise ValueError("SHARED_GO_fig4_canonical_30terms and S6_GO_matrix disagree on "
                         "the 30 Term strings")

    m = out["membership"]
    m["program"] = m["program"].map(_key)
    got = dict(zip(m.program, m.n_genes.astype(int)))
    if got != DECK_WHOLE_SIZES:
        raise ValueError(f"S3d membership sizes {got} disagree with the deck "
                         f"{DECK_WHOLE_SIZES}")
    return out


def _bh(p):
    """Benjamini-Hochberg over one contrast's tested genes."""
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.clip(q, 0.0, 1.0)
    return out


def volcano_stats(counts):
    """Per-gene control-anchored statistics: counts -> CPM -> log2(x+1) -> Welch t.

    CTRL_3 failed QC and is already absent, so control is n=2 and every stage n=3. Each
    contrast is tested on the genes with a nonzero count in at least one of its own five
    samples; that per-contrast detection filter is what reproduces MD_a's four n_tested
    values (16,652 / 16,642 / 17,174 / 17,463). A single global filter cannot, because it
    yields one gene count for all four contrasts.
    """
    cpm = counts / counts.sum(0) * 1e6
    lg = np.log2(cpm + 1.0)
    groups = {s: [c for c in counts.columns if c.rsplit("_", 1)[0] == s] for s in STAGES}
    detected = counts > 0
    out = {}
    for stage, _ in CONTRASTS:
        cols = groups["CTRL"] + groups[stage]
        keep = detected[cols].sum(1) >= 1
        a = lg.loc[keep, groups[stage]].values
        b = lg.loc[keep, groups["CTRL"]].values
        p = stats.ttest_ind(a, b, axis=1, equal_var=False).pvalue
        p = np.where(np.isnan(p), 1.0, p)
        out[stage] = pd.DataFrame({"gene": lg.index[keep], "log2FC": a.mean(1) - b.mean(1),
                                   "p": p, "q": _bh(p)})
    return out


def _assert_volcano(res, volc_counts):
    """Stop the build unless the recomputed clouds carry MD_a's counts exactly."""
    v = volc_counts.set_index("contrast")
    for stage, row in CONTRASTS:
        d = res[stage]
        sig = (d.q < Q_CUT) & (d.log2FC.abs() >= FC_CUT)
        got = (int((sig & (d.log2FC < 0)).sum()), int((sig & (d.log2FC > 0)).sum()),
               len(d))
        want = (int(v.loc[row, "n_down_control_high"]), int(v.loc[row, "n_up_stage_high"]),
                int(v.loc[row, "n_tested"]))
        if got != want:
            raise ValueError(
                f"{row}: recomputed (control-high, stage-high, tested) = {got}, "
                f"MD_a gives {want}. The recipe is not adjusted to close this.")


def panel_a(d):
    """Four control-anchored volcanoes on shared axes."""
    res = volcano_stats(d["counts"])
    _assert_volcano(res, d["volc_counts"])
    fig, axes = plt.subplots(2, 2, figsize=(7.3, 6.4), sharex=True, sharey=False)
    xlo = min(r.log2FC.min() for r in res.values()) - 0.6
    xhi = max(r.log2FC.max() for r in res.values()) + 0.6
    for ax, (stage, _) in zip(axes.ravel(), CONTRASTS):
        prog = [p for p in PROGRAMS if PROG_PEAK[p] == stage][0]
        col = MOD_COL[prog]
        t = res[stage].copy()
        t["y"] = -np.log10(np.maximum(t.q, 1e-300))
        sig = (t.q < Q_CUT) & (t.log2FC.abs() >= FC_CUT)
        up, dn = sig & (t.log2FC > 0), sig & (t.log2FC < 0)
        ax.axhline(-np.log10(Q_CUT), color="#9a9a9a", lw=0.5, ls=(0, (3, 2)), zorder=1)
        for x in (-FC_CUT, FC_CUT):
            ax.axvline(x, color="#9a9a9a", lw=0.5, ls=(0, (3, 2)), zorder=1)
        ax.scatter(t.log2FC[~sig], t.y[~sig], s=1.6, c=GREY_NS, lw=0, zorder=2)
        ax.scatter(t.log2FC[dn], t.y[dn], s=2.6, c=GREY_SIG, lw=0, zorder=3)
        ax.scatter(t.log2FC[up], t.y[up], s=2.6, c=col, lw=0, zorder=4)
        ax.set_xlim(xlo, xhi)
        ax.set_ylim(-0.12, t.y.max() * 1.30)
        ax.set_title(f"Control vs {STAGE_LAB[stage]}", fontsize=8.4, loc="left",
                     color=col, fontweight="bold", pad=9)
        ax.text(0.02, 0.965, f"{int(dn.sum()):,} control-high", transform=ax.transAxes,
                fontsize=7.0, color=GREY_SIG, ha="left", va="top")
        ax.text(0.98, 0.965, f"{int(up.sum()):,} {STAGE_LAB[stage]}-high",
                transform=ax.transAxes, fontsize=7.0, color=col, ha="right", va="top")
        # Top four stage-high genes by raw P. Within a BH plateau the P order is the only
        # ordering left, so this set is close to but not identical with the delivered one.
        lab = t[up].nsmallest(4, "p")
        for i, (_, g) in enumerate(lab.iterrows()):
            ax.annotate(g.gene, xy=(g.log2FC, g.y),
                        xytext=(0.995, 0.80 - 0.135 * i), textcoords=ax.transAxes,
                        fontsize=6.4, style="italic", color=INK, ha="right", va="center",
                        arrowprops=dict(arrowstyle="-", lw=0.4, color="#8a8a8a",
                                        shrinkA=1.5, shrinkB=1.5))
    for ax in axes[1]:
        ax.set_xlabel("log$_2$ fold change vs control", fontsize=7.4)
    for ax in axes[:, 0]:
        ax.set_ylabel("$-$log$_{10}$ adjusted $P$", fontsize=7.4)
    fig.text(0.5, 0.005, "Grey = higher in control; control-high in at least 3 of the 4 "
             "contrasts defines the Homeostasis program.",
             fontsize=6.6, color="#777777", ha="center", va="bottom")
    fig.subplots_adjust(left=0.085, right=0.985, top=0.93, bottom=0.10,
                        wspace=0.20, hspace=0.30)
    return fig


def panel_b(d):
    """Observed program mean z at every stage against the permutation null 95% band."""
    n = d["null"].copy()
    fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.3), sharey=True)
    x = np.arange(5)
    for ax, prog in zip(axes, PROGRAMS):
        t = n[n.program == prog].set_index("stage").loc[STAGES]
        col = MOD_COL[prog]
        ax.fill_between(x, t.null_p2_5.values, t.null_p97_5.values, color=GREY_NULL,
                        alpha=0.55, lw=0, zorder=1, label="null 95%")
        ax.axhline(0.0, color="#c4c4c4", lw=0.5, zorder=1)
        ax.plot(x, t.observed_mean_z.values, color=col, lw=1.3, zorder=3)
        inside = t.inside_null_95.values.astype(bool)
        ax.scatter(x[~inside], t.observed_mean_z.values[~inside], s=22, c=col,
                   edgecolors="white", linewidths=0.5, zorder=4)
        ax.scatter(x[inside], t.observed_mean_z.values[inside], s=22, facecolors="white",
                   edgecolors=col, linewidths=0.9, zorder=4)
        pk = t[t.is_peak_stage.astype(bool)].iloc[0]
        ax.set_title(disp(prog), fontsize=7.6, color=col, loc="left", pad=7)
        ax.text(0.0, 1.005, f"{int(pk.n_genes_used)}/{int(pk.n_genes_program)} genes "
                f"\u00b7 {pk.z_vs_null:.0f} SD", transform=ax.transAxes,
                fontsize=5.9, color="#777777", ha="left", va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels([STAGE_LAB[s] for s in STAGES], rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=6.2, color="#555555")
        ax.set_xlim(-0.45, 4.45)
    axes[0].set_ylabel("mean gene $z$", fontsize=7.4)
    h = [plt.Line2D([], [], marker="o", ls="none", mfc=INK, mec="white", ms=4.4),
         plt.Line2D([], [], marker="o", ls="none", mfc="white", mec=INK, ms=4.4),
         Rectangle((0, 0), 1, 1, fc=GREY_NULL, alpha=0.55, lw=0)]
    fig.legend(h, ["outside null 95%", "inside null 95%", "null 95% band"],
               loc="lower center", ncol=3, fontsize=6.4, bbox_to_anchor=(0.5, -0.055),
               handletextpad=0.5, columnspacing=1.6)
    fig.text(0.5, -0.15, "Null: random gene sets per program matched on set size and "
             "control-expression decile. Homeostasis at PPD7 is the one cell of 25 "
             "inside the band.", fontsize=6.2, color="#777777", ha="center", va="bottom")
    fig.subplots_adjust(left=0.075, right=0.995, top=0.80, bottom=0.24, wspace=0.14)
    return fig


def _overlap_matrices(ov):
    idx = {p: i for i, p in enumerate(PROGRAMS)}
    cnt = np.zeros((5, 5))
    jac = np.zeros((5, 5))
    for r in ov.itertuples():
        i, j = idx[r.program_a], idx[r.program_b]
        cnt[i, j] = r.intersection
        jac[i, j] = r.jaccard
    return cnt, jac


def panel_c(d):
    """Shared-gene counts and Jaccard percentages between the five programs."""
    cnt, jac = _overlap_matrices(d["overlap"])
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.5))
    labs = [disp(p) for p in PROGRAMS]
    for ax, M, title, fmt in (
            (axes[0], cnt, "Shared genes", "count"),
            (axes[1], jac * 100.0, "Jaccard (% of the union)", "pct")):
        off = np.array(M, float)
        np.fill_diagonal(off, np.nan)
        im = ax.imshow(np.ma.masked_invalid(off), cmap="viridis", origin="upper")
        for i in range(5):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fc=MOD_COL[PROGRAMS[i]],
                                   ec="white", lw=0.8, zorder=2))
        for i in range(5):
            for j in range(5):
                v = M[i, j]
                if fmt == "count":
                    s = f"{int(round(v)):,}"
                else:
                    s = "0" if (i != j and v == 0) else f"{v:.1f}"
                on_diag = i == j
                # viridis runs dark purple to yellow, so the LOW cells need white text.
                pale = (not on_diag) and v > 0.55 * float(np.nanmax(off))
                ax.text(j, i, s, ha="center", va="center", fontsize=6.4,
                        color=INK if pale else "white",
                        fontweight="bold" if on_diag else "normal", zorder=3)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(labs, rotation=45, ha="right", rotation_mode="anchor",
                           fontsize=6.4)
        ax.set_yticklabels(labs, fontsize=6.4)
        for i, p in enumerate(PROGRAMS):
            ax.get_xticklabels()[i].set_color(MOD_COL[p])
            ax.get_yticklabels()[i].set_color(MOD_COL[p])
        ax.set_title(title, fontsize=7.8, loc="left", pad=8)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=5.8, length=2)
        cb.outline.set_visible(False)
    a, b, k, j = DECK_TOP_OVERLAP
    fig.text(0.5, 0.005, f"Diagonal = program size, filled with the program colour. "
             f"Largest overlap {disp(a)} \u2229 {disp(b)} = {k} genes "
             f"(Jaccard {j * 100:.1f}%). Homeostasis \u2229 Early repair = 0 "
             f"by construction.", fontsize=6.4, color="#777777", ha="center", va="bottom")
    fig.subplots_adjust(left=0.135, right=0.985, top=0.90, bottom=0.20, wspace=0.55)
    return fig


def panel_d(d):
    """One program heatmap per compartment, shared row order, shared colour scale."""
    blocks, sizes = [], {}
    start = 0
    ref = d["prog_Epithelial"].sort_values("row_index")
    for p in PROGRAMS:
        n = int((ref.program == p).sum())
        blocks.append((p, start, start + n))
        sizes[p] = n
        start += n
    if sizes != DECK_COMP_SIZES:
        raise ValueError(f"panel d block sizes {sizes} disagree with the deck "
                         f"{DECK_COMP_SIZES}")
    zcols = ["z_" + s for s in STAGES]
    mats, tabs = {}, {}
    for c in COMPS:
        t = d["prog_" + c].sort_values("row_index").reset_index(drop=True)
        if set(t.gene) != set(ref.gene):
            raise ValueError(f"{c} carries a different gene set from Epithelial")
        got = {p: int((t.program == p).sum()) for p in PROGRAMS}
        if got != sizes:
            raise ValueError(f"{c} block sizes {got} differ from Epithelial {sizes}")
        mats[c] = t[zcols].values
        tabs[c] = t

    lim = float(np.nanpercentile(np.abs(np.concatenate(list(mats.values()))), 99.5))
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 5.0))
    for ax, c in zip(axes, COMPS):
        im = ax.imshow(mats[c], cmap="RdBu_r", norm=norm, aspect="auto",
                       interpolation="nearest")
        for _, s, e in blocks[:-1]:
            ax.axhline(e - 0.5, color="white", lw=1.1)
        ax.set_xticks(range(5))
        ax.set_xticklabels([STAGE_LAB[s] for s in STAGES], rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=6.2, color="#555555")
        ax.set_yticks([])
        ax.set_title(c, fontsize=7.8, color=COMP_COL[c], loc="left", pad=7)
        for sp in ax.spines.values():
            sp.set_visible(False)
    # The tables carry a `labelled` flag naming 5-6 genes per compartment. Those callouts
    # are not drawn: four panels 25 mm wide leave no room for a leader line that does not
    # cross the next panel, and the program blocks are what this panel is read for.
    for p, s, e in blocks:
        axes[0].add_patch(Rectangle((-0.95, s - 0.5), 0.35, e - s, fc=MOD_COL[p],
                                    ec="none", clip_on=False, zorder=3))
        axes[0].text(-1.25, (s + e) / 2.0, f"{disp(p)}\nn={e - s}", ha="right",
                     va="center", fontsize=6.4, color=MOD_COL[p], clip_on=False)
    cb = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02)
    cb.set_label("gene $z$ across stages, within compartment", fontsize=6.6)
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_visible(False)
    fig.text(0.5, 0.005, "The same 2,935 genes and the same five program blocks in every "
             "compartment, each compartment z-scored inside itself and ordered inside "
             "each block by its own table. Rows do not correspond across the four panels.",
             fontsize=6.3, color="#777777", ha="center", va="bottom")
    fig.subplots_adjust(left=0.215, right=0.90, top=0.93, bottom=0.115, wspace=0.10)
    return fig


def panel_e(d):
    """The 30 Fig 4 GO terms against all 20 compartment modules."""
    g = d["go"].copy()
    rows = d["go_rows"].copy()
    rows = rows.sort_values("program", key=lambda s: s.map(PROGRAMS.index)).reset_index(
        drop=True)
    yof = {t: i for i, t in enumerate(rows.Term)}
    cols = [(c, m) for c in COMPS for m in (1, 2, 3, 4, 5)]
    xof = {cm: i for i, cm in enumerate(cols)}
    meta = g.groupby(["compartment", "module"]).agg(
        cl=("column_label", "first"), ms=("module_size", "first"))

    t = g[g.tested.astype(bool) & g.k.gt(0)].copy()
    t["x"] = list(zip(t.compartment, t.module))
    t["x"] = t["x"].map(xof)
    t["y"] = t.Term.map(yof)
    fig, ax = plt.subplots(figsize=(7.4, 7.6))
    for p in PROGRAMS:
        ii = rows.index[rows.program == p]
        ax.add_patch(Rectangle((-0.6, ii.min() - 0.5), len(cols) + 0.2 - 0.4 + 0.4,
                               len(ii), fc=MOD_COL[p], alpha=0.07, lw=0, zorder=0))
        ax.text(-0.9, (ii.min() + ii.max()) / 2.0, disp(p), rotation=90, ha="center",
                va="center", fontsize=6.6, color=MOD_COL[p])
    smax = float(t.k.max())
    sc = ax.scatter(t.x, t.y, s=8 + 150 * t.k / smax, c=t.negLog10q, cmap="viridis",
                    vmin=0.0, vmax=float(t.negLog10q.max()), lw=0.25,
                    edgecolors="#ffffff", zorder=3)
    sg = t[t.sig.astype(bool)]
    ax.scatter(sg.x, sg.y, s=8 + 150 * sg.k / smax, facecolors="none",
               edgecolors="#c0392b", linewidths=0.9, zorder=4)
    for i in range(1, 4):
        ax.axvline(5 * i - 0.5, color="#dddddd", lw=0.7, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows.label, fontsize=6.2, color=INK)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"M{m} \u00b7 {meta.loc[(c, m), 'cl']}" for c, m in cols],
                       rotation=45, ha="right", rotation_mode="anchor", fontsize=5.9)
    ax.set_xlim(-1.15, len(cols) - 0.4)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i, c in enumerate(COMPS):
        ax.text(5 * i + 2, -1.15, c, ha="center", va="bottom", fontsize=7.4,
                color=COMP_COL[c])
    cb = fig.colorbar(sc, ax=ax, fraction=0.030, pad=0.015)
    cb.set_label("$-$log$_{10}$ q", fontsize=6.6)
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_visible(False)
    n_sig = int(t.sig.astype(bool).sum())
    n_terms = int(sg.Term.nunique())
    ax.set_title("Whole-tissue program terms against each compartment's own modules",
                 fontsize=8.0, loc="left", pad=26)
    fig.text(0.5, 0.005, f"Dot size = genes of the term in that module; red ring = "
             f"q < 0.05 ({n_sig} of {len(t)} tested cells, {n_terms} of 30 terms). No dot "
             f"= the term has no gene in that module or is absent from that "
             f"compartment's universe.", fontsize=6.3, color="#777777", ha="center",
             va="bottom")
    fig.subplots_adjust(left=0.315, right=0.925, top=0.925, bottom=0.145)
    return fig


_PANELS = [("a", panel_a, "SFig3A_volcanoes_ctrl_anchored"),
           ("b", panel_b, "SFig3B_program_null_bands"),
           ("c", panel_c, "SFig3C_program_overlap"),
           ("d", panel_d, "SFig3D_program_heatmaps_by_compartment"),
           ("e", panel_e, "SFig4A_program_GO_matrix_4compartments")]


def build(repo_root: str, out_dir: str, letters=None) -> dict:
    """Render every Fig S3 panel. Returns {panel letter: svg path}."""
    d = load_inputs(repo_root)
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    out = {}
    for letter, fn, stem in _PANELS:
        if letters is not None and letter not in letters:
            continue
        path = os.path.join(out_dir, stem)
        fig = fn(d)
        render_svg.save_svg(fig, path)
        if letter == "a":
            # Four volcanoes are ~68,000 live markers. compress_svg is lossless and keeps
            # every point editable, which rasterizing the cloud would not.
            compress_svg(path + ".svg")
        info = render_svg.verify(path + ".svg")
        if not info["text_elements"] or not info["named_font"]:
            raise RuntimeError(f"{stem}.svg exported without live text: {info}")
        plt.close(fig)
        out[letter] = path + ".svg"
    return out
