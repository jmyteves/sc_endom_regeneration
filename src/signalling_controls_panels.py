"""Communication-inference controls and the epithelial ligand-receptor detail (fig. S5, extras).

    build(repo_root, out_dir, letters=None) -> {letter: svg_path}

  a  inferred interaction totals against cells sequenced per stage, one axes per
     measure with its Spearman rho and p. Four measures are drawn; the delivered panel
     showed three and left out n_tested (rho -0.70, p = 0.188), which is marked as such.
  b  cells available per cell type and stage, 16 x 5, the one combination excluded from
     inference hatched. Colour is log-scaled (range 8 to 233,000 cells).
  c  ligand-receptor pairs for the two principal compartment pairs in both directions,
     14 pairs per direction. Colour is the control share of the pair's signal (0.5 =
     control and stage contribute equally); dot size the total signal.   (fig. S5 A-D)
  d  the epithelial share of signal resolved to individual ligands, epithelium as
     receiver and as sender; rows ordered by mean control share.
  e  Jaccard similarity of the active interaction set to control per stage.

Only c is in the v10 supplementary set; a, b, d and e build under extras_panels.
CTRL is the reference denominator in c and d, not a plotted stage, so those panels
carry four stage columns while a and b carry five. Stages take the paper stage palette;
diverging control share keeps PuOr_r centred on 0.5 and cell counts keep viridis.
"""
import os

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle

import render_svg

STAGES5 = ["CTRL", "E16_5", "PPD1", "PPD7", "PPD30"]
STAGES4 = ["E16_5", "PPD1", "PPD7", "PPD30"]
STAGE_LAB = {"CTRL": "Control", "E16_5": "E16.5", "PPD1": "PPD1", "PPD7": "PPD7",
             "PPD30": "PPD30"}
STAGE_COL = {"CTRL": "#808080", "E16_5": "#cb7a29", "PPD1": "#A0522D",
             "PPD7": "#FA8072", "PPD30": "#4682B4"}
COMPS = ["Epithelial", "Stromal", "Endothelial", "Immune"]
COMP_COL = {"Epithelial": "#748190", "Stromal": "#367f80",
            "Endothelial": "#e58677", "Immune": "#c58a4c"}
COMP_SHORT = {"Epithelial": "Epi", "Stromal": "Stromal", "Endothelial": "Endo",
              "Immune": "Immune"}
INK = "#333333"
ALARM = "#c0392b"

# The four depth measures, in the order the stats table lists them, with the axis label
# each one needs and whether the delivered panel plots it.
MEASURES = [
    ("n_active_p05", "Active LR interactions\n($p$ < 0.05, 16 cell types)", True),
    ("published_load_4comp", "Published signalling load\n(4 compartments)", True),
    ("edge_total_16types", "Total directed edges\n(16 x 16 type pairs)", True),
    ("n_tested", "LR pairs tested\n(not in the delivered panel)", False),
]

# Deck values. Asserted, not fitted: a disagreement stops the build.
DECK_SPEARMAN = {"n_active_p05": (-0.6, 0.2848), "published_load_4comp": (-0.9, 0.0374),
                 "edge_total_16types": (-0.6, 0.2848), "n_tested": (-0.7, 0.1881)}
DECK_EXCLUSION = ("E16_5", "Stromal", "Fibroblast (Igf1+)", 8, "dropped (<10 cells)")
DECK_N_STAGES = 5

_TABLES = {
    "depth": "S5a_interactions_vs_depth_source.csv",
    "depth_stats": "S5a_interactions_vs_depth_stats.csv",
    "coverage": "S5b_coverage_source.csv",
    "lr": "S5c_LR_dotplots_principal_pairs_source.csv",
    "share": "S5e_epithelial_LR_share_source.csv",
    "share_labels": "S5e_epithelial_LR_share_delivered_end_labels.csv",
    "rewiring": "S5f_rewiring_vs_control_source.csv",
}


def _read(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Fig S4 input missing: {path}. Stage it from "
            f"sc_endom_EDITABLE_SVG/SFigs_latest/tables/{os.path.basename(path)}")
    return pd.read_csv(path)


def load_inputs(repo_root):
    """Read and check every staged table. Raises before any drawing on a disagreement."""
    d = os.path.join(repo_root, "data", "raw_light", "sfig4")
    out = {k: _read(os.path.join(d, fn)) for k, fn in _TABLES.items()}

    dep = out["depth"]
    if list(dep.stage) != STAGES5:
        raise ValueError(f"S5a source stages {list(dep.stage)}, expected {STAGES5}")
    st = out["depth_stats"].set_index("measure")
    for m, _, _ in MEASURES:
        if m not in st.index:
            raise ValueError(f"S5a stats has no row for {m}")
        rho, p = DECK_SPEARMAN[m]
        got = (round(float(st.loc[m, "spearman_rho"]), 4),
               round(float(st.loc[m, "spearman_p"]), 4))
        if got != (rho, p):
            raise ValueError(f"S5a stats {m}: rho/p {got} disagrees with the deck "
                             f"{(rho, p)}")
        if int(st.loc[m, "n_stages"]) != DECK_N_STAGES:
            raise ValueError(f"S5a stats {m}: n_stages {st.loc[m, 'n_stages']}, "
                             f"expected {DECK_N_STAGES}")

    cov = out["coverage"]
    if len(cov) != 80 or cov.cell_type.nunique() != 16 or cov.stage.nunique() != 5:
        raise ValueError(f"S5b: {len(cov)} rows over {cov.cell_type.nunique()} types x "
                         f"{cov.stage.nunique()} stages, expected 80 over 16 x 5")
    drop = cov[cov.status != "tested"]
    if len(drop) != 1:
        raise ValueError(f"S5b: {len(drop)} non-tested combinations, expected exactly 1")
    r = drop.iloc[0]
    got = (r.stage, r.compartment, r.cell_type, int(r.n_cells), r.status)
    if got != DECK_EXCLUSION:
        raise ValueError(f"S5b exclusion {got} disagrees with the deck {DECK_EXCLUSION}")

    lr = out["lr"]
    if lr.direction.nunique() != 4:
        raise ValueError(f"S5c: {lr.direction.nunique()} directions, expected 4")
    for dr, g in lr.groupby("direction"):
        if g.lr_pair.nunique() != 14:
            raise ValueError(f"S5c {dr}: {g.lr_pair.nunique()} pairs, expected 14")
        if g.groupby("lr_pair").plot_rank_from_bottom.nunique().max() != 1:
            raise ValueError(f"S5c {dr}: plot_rank_from_bottom varies within a pair")
    if sorted(lr.stage.unique()) != sorted(STAGES4):
        raise ValueError(f"S5c stages {sorted(lr.stage.unique())}, expected {STAGES4}")

    sh = out["share"]
    if sorted(sh.role.unique()) != ["receiver", "sender"]:
        raise ValueError(f"S5e roles {sorted(sh.role.unique())}, expected receiver/sender")
    if sorted(sh.stage.unique()) != sorted(STAGES4):
        raise ValueError(f"S5e stages {sorted(sh.stage.unique())}, expected {STAGES4}")
    lab = out["share_labels"]
    for role in ("receiver", "sender"):
        want = set()
        for s in lab[lab.role == role].ligands:
            want |= {x.strip() for x in str(s).split(",")}
        have = set(sh[sh.role == role].ligand)
        missing = sorted(want - have)
        if missing:
            raise ValueError(f"S5e {role}: delivered end labels {missing} are absent from "
                             "the plotted ligand set")

    rw = out["rewiring"]
    if sorted(rw.stage) != sorted(STAGES4):
        raise ValueError(f"S5f stages {sorted(rw.stage)}, expected {sorted(STAGES4)}")
    if rw.n_ctrl.nunique() != 1:
        raise ValueError(f"S5f: {rw.n_ctrl.nunique()} distinct control edge totals, "
                         "expected one shared anchor")
    bad = rw[rw.gained + rw.shared != rw.n_stage]
    if len(bad):
        raise ValueError("S5f: gained + shared != n_stage for "
                         f"{list(bad.stage)}; the edge sets do not close")
    bad = rw[rw.lost + rw.shared != rw.n_ctrl]
    if len(bad):
        raise ValueError("S5f: lost + shared != n_ctrl for "
                         f"{list(bad.stage)}; the edge sets do not close")
    j = rw.set_index("stage").jaccard
    recomputed = rw.set_index("stage").eval("shared / (shared + gained + lost)")
    off = (j - recomputed).abs().max()
    if off > 5e-4:
        raise ValueError(f"S5f: stored jaccard disagrees with shared/(shared+gained+lost) "
                         f"by up to {off:.4f}")
    return out


def _fmt_k(v):
    """Cell counts as the deck writes them: 8, 559, 4.7k, 233k."""
    v = float(v)
    if v < 1000:
        return f"{int(round(v))}"
    if v < 10000:
        return f"{v / 1000:.1f}k"
    return f"{v / 1000:.0f}k"


def panel_a(d):
    """Each depth measure against cells sequenced, with its Spearman coefficient."""
    dep = d["depth"].set_index("stage").loc[STAGES5]
    st = d["depth_stats"].set_index("measure")
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.4))
    x = dep.n_cells.values / 1000.0
    for ax, (m, lab, plotted) in zip(axes, MEASURES):
        y = dep[m].values.astype(float)
        for xi, yi, s in zip(x, y, STAGES5):
            ax.scatter([xi], [yi], s=46, c=STAGE_COL[s], lw=0.5, edgecolors="white",
                       zorder=3)
        rho = float(st.loc[m, "spearman_rho"])
        p = float(st.loc[m, "spearman_p"])
        ax.set_title(f"$\\rho$ = {rho:.2f}  ($p$ = {p:.2f})", fontsize=7.2,
                     color=INK if plotted else "#8a8a8a", loc="left", pad=6)
        ax.set_ylabel(lab, fontsize=6.6, color=INK if plotted else "#8a8a8a")
        ax.margins(0.18)
        ax.tick_params(labelsize=6.0)
        if not plotted:
            for sp in ax.spines.values():
                sp.set_color("#9a9a9a")
                sp.set_linestyle((0, (2.5, 2)))
    fig.text(0.5, 0.135, "Cells sequenced per stage (thousands)", fontsize=7.2,
             ha="center", va="bottom")
    h = [plt.Line2D([], [], marker="o", ls="none", mfc=STAGE_COL[s], mec="white", ms=5.0)
         for s in STAGES5]
    fig.legend(h, [STAGE_LAB[s] for s in STAGES5], loc="lower center", ncol=5,
               fontsize=6.6, bbox_to_anchor=(0.5, 0.005), handletextpad=0.4,
               columnspacing=1.4)
    fig.suptitle("Inferred interaction totals do not track sequencing depth "
                 "(Spearman, n = 5 stages)", fontsize=8.0, x=0.008, ha="left")
    fig.subplots_adjust(left=0.085, right=0.99, top=0.80, bottom=0.28, wspace=0.62)
    return fig


def panel_b(d):
    """The 16 x 5 cells-available matrix with the single exclusion hatched."""
    cov = d["coverage"].copy()
    order = (cov[["compartment", "cell_type"]].drop_duplicates()
             .assign(k=lambda t: t.compartment.map(COMPS.index))
             .sort_values(["k", "cell_type"]))
    types = list(order.cell_type)
    comp_of = dict(zip(order.cell_type, order.compartment))
    M = (cov.pivot(index="cell_type", columns="stage", values="n_cells")
         .loc[types, STAGES5].values.astype(float))
    status = cov.pivot(index="cell_type", columns="stage", values="status").loc[
        types, STAGES5].values

    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    im = ax.imshow(M, cmap="viridis", norm=LogNorm(vmin=M.min(), vmax=M.max()),
                   aspect="auto")
    for i in range(len(types)):
        for j in range(5):
            dropped = status[i, j] != "tested"
            if dropped:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fc="none", ec=ALARM,
                                       lw=1.2, hatch="////", zorder=3))
            # White on every cell: the excluded cell's red hatch sits over a dark fill,
            # and red-on-red made its count the least readable number in the panel.
            ax.text(j, i, _fmt_k(M[i, j]), ha="center", va="center", fontsize=6.4,
                    color="white", fontweight="bold" if dropped else "normal", zorder=4)
    ax.set_xticks(range(5))
    ax.set_xticklabels([STAGE_LAB[s] for s in STAGES5], fontsize=7.0)
    ax.set_yticks(range(len(types)))
    ax.set_yticklabels(types, fontsize=6.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Compartment spine to the left of the row labels, one bar per run of rows.
    runs = []
    for i, t in enumerate(types):
        c = comp_of[t]
        if runs and runs[-1][0] == c:
            runs[-1][2] = i
        else:
            runs.append([c, i, i])
    for c, i0, i1 in runs:
        ax.add_patch(Rectangle((-2.28, i0 - 0.44), 0.18, i1 - i0 + 0.88,
                               fc=COMP_COL[c], ec="none", clip_on=False, zorder=3))
        ax.text(-2.19, (i0 + i1) / 2.0, COMP_SHORT[c], rotation=90, ha="center",
                va="center", fontsize=6.0, color="white", clip_on=False, zorder=4)
    cb = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.02)
    cb.set_label("Cells available (log scale)", fontsize=6.8)
    cb.ax.tick_params(labelsize=6.0, length=2)
    cb.outline.set_visible(False)
    st, comp, ct, n, why = DECK_EXCLUSION
    ax.set_title("All 16 cell types tested at all 5 stages except one combination",
                 fontsize=8.0, loc="left", pad=10)
    fig.text(0.5, 0.012, f"Hatched = excluded from inference: {STAGE_LAB[st]} / {ct} "
             f"({comp}), n = {n} cells, {why}. The other 79 combinations were tested.",
             fontsize=6.6, color="#777777", ha="center", va="bottom")
    fig.subplots_adjust(left=0.375, right=0.90, top=0.945, bottom=0.09)
    return fig


def _share_dots(ax, t, rows, stages, size_ref, cmap="PuOr_r", rot=0):
    """One share dotplot: rows x stages, colour = control share, size = total signal."""
    yof = {r: i for i, r in enumerate(rows)}
    xof = {s: i for i, s in enumerate(stages)}
    norm = TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0)
    for i in range(len(rows)):
        ax.axhline(i, color="#f0f0f0", lw=0.6, zorder=0)
    present = set()
    sc = None
    for r in t.itertuples():
        present.add((r.row, r.stage))
        sc = ax.scatter([xof[r.stage]], [yof[r.row]],
                        s=10 + 190 * r.total_signal / size_ref, c=[r.control_share],
                        cmap=cmap, norm=norm, lw=0.35, edgecolors="#666666", zorder=3)
    for r in rows:
        for s in stages:
            if (r, s) not in present:
                ax.scatter([xof[s]], [yof[r]], s=6, facecolors="none",
                           edgecolors="#999999", linewidths=0.4, zorder=3)
    ax.set_xticks(range(len(stages)))
    ax.set_xticklabels([STAGE_LAB[s] for s in stages], fontsize=6.8, rotation=rot,
                       ha="right" if rot else "center",
                       rotation_mode="anchor" if rot else None)
    ax.set_yticks(range(len(rows)))
    ax.set_xlim(-0.6, len(stages) - 0.4)
    ax.set_ylim(len(rows) - 0.5, -0.5)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    return sc, norm


def panel_c(d):
    """Ligand-receptor detail for the two principal compartment pairs, both directions."""
    lr = d["lr"].copy()
    lr["row"] = lr.lr_pair
    dirs = ["Stromal->Epithelial", "Epithelial->Stromal",
            "Immune->Epithelial", "Epithelial->Immune"]
    size_ref = float(lr.total_signal.max())
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 4.6))
    sc = None
    for ax, dr in zip(axes, dirs):
        t = lr[lr.direction == dr]
        rows = list(t.drop_duplicates("lr_pair")
                    .sort_values("plot_rank_from_bottom", ascending=False).lr_pair)
        sc, _ = _share_dots(ax, t, rows, STAGES4, size_ref, rot=45)
        sender = dr.split("->")[0]
        ax.set_title(dr.replace("->", " \u2192 "), fontsize=7.4, loc="left", pad=8,
                     color=COMP_COL[sender])
        ax.set_yticklabels([p.replace("|", " \u2192 ").replace("_", " + ")
                            for p in rows], fontsize=6.0)
    cb = fig.colorbar(sc, ax=axes, fraction=0.014, pad=0.012)
    cb.set_label("Control share of the pair's signal\n(0 = stage-specific, "
                 "1 = control-dominated)", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_visible(False)
    sizes = [round(size_ref / 8), round(size_ref / 2), round(size_ref)]
    h = [plt.Line2D([], [], marker="o", ls="none", mfc="#b0b0b0", mec="#666666",
                    ms=np.sqrt(10 + 190 * s / size_ref)) for s in sizes]
    h.append(plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#999999",
                        ms=np.sqrt(6)))
    fig.legend(h, [f"{s:g}" for s in sizes] + ["not detected at that stage"],
               title="Total signal (control + stage)", title_fontsize=6.4,
               loc="lower center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.45, -0.02),
               handletextpad=0.5, columnspacing=1.4)
    fig.suptitle("Ligand-receptor detail for the principal compartment pairs, both "
                 "directions", fontsize=8.0, x=0.008, ha="left")
    fig.text(0.5, -0.075, "Top 14 pairs per direction by signal summed over stages, "
             "aggregated over the constituent cell-type pairs. Control is the reference "
             "for every share, not a plotted stage.", fontsize=6.3, color="#777777",
             ha="center", va="bottom")
    fig.subplots_adjust(left=0.155, right=0.885, top=0.90, bottom=0.155, wspace=1.35)
    return fig


def panel_d(d):
    """Epithelial share of signal per ligand, epithelium as receiver and as sender."""
    sh = d["share"].copy()
    sh["row"] = sh.ligand
    lab = d["share_labels"]
    size_ref = float(sh.total_signal.max())
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 5.4))
    titles = {"receiver": "Epithelium as RECEIVER (all other types \u2192 epithelium)",
              "sender": "Epithelium as SENDER (epithelium \u2192 all other types)"}
    sc = None
    n_named = {}
    for ax, role in zip(axes, ("receiver", "sender")):
        t = sh[sh.role == role]
        # Ordering by mean control share reproduces the delivered row order in both roles:
        # control-dominated ligands at the top, stage-specific at the bottom.
        rows = list(t.groupby("ligand").control_share.mean()
                    .sort_values(ascending=False).index)
        sc, _ = _share_dots(ax, t, rows, STAGES4, size_ref)
        ax.set_yticklabels(rows, fontsize=6.6, style="italic")
        ax.set_title(titles[role], fontsize=7.2, loc="left", pad=8,
                     color=COMP_COL["Epithelial"])
        named = set()
        for s in lab[lab.role == role].ligands:
            named |= {x.strip() for x in str(s).split(",")}
        # The footer claims every delivered end label is among the plotted rows, so the
        # claim is computed here rather than assumed from load_inputs having passed.
        outside = sorted(named - set(rows))
        if outside:
            raise ValueError(f"S5e {role}: delivered end labels {outside} are not among "
                             "the plotted rows, so the panel footer would be false")
        n_named[role] = (len(named & set(rows)), len(rows))
    cb = fig.colorbar(sc, ax=axes, fraction=0.017, pad=0.015)
    cb.set_label("Control share of ligand signal\n(0 = stage-specific, "
                 "1 = control-dominated)", fontsize=6.2)
    cb.ax.tick_params(labelsize=5.8, length=2)
    cb.outline.set_visible(False)
    sizes = [round(size_ref / 8), round(size_ref / 2), round(size_ref)]
    h = [plt.Line2D([], [], marker="o", ls="none", mfc="#b0b0b0", mec="#666666",
                    ms=np.sqrt(10 + 190 * s / size_ref)) for s in sizes]
    h.append(plt.Line2D([], [], marker="o", ls="none", mfc="none", mec="#999999",
                        ms=np.sqrt(6)))
    fig.legend(h, [f"{s:g}" for s in sizes] + ["not detected at that stage"],
               title="Total signal (control + stage)", title_fontsize=6.4,
               loc="lower center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.45, -0.015),
               handletextpad=0.5, columnspacing=1.4)
    fig.suptitle("Epithelial share of signal resolved to individual ligands",
                 fontsize=8.0, x=0.008, ha="left")
    fig.text(0.5, -0.068, f"Rows are ordered by mean control share. Every ligand the "
             f"delivered end-label table names is inside the plotted set "
             f"({n_named['receiver'][0]} of {n_named['receiver'][1]} rows as receiver, "
             f"{n_named['sender'][0]} of {n_named['sender'][1]} as sender). Control is "
             f"the reference denominator, not a plotted stage: 4 stages x 2 roles.",
             fontsize=6.3, color="#777777", ha="center", va="bottom")
    fig.subplots_adjust(left=0.105, right=0.885, top=0.905, bottom=0.135, wspace=0.42)
    return fig


def panel_e(d):
    """Edge-set similarity to control, and the gained/lost edges behind it.

    Fig 6 asserts in its title that the network is least like the non-pregnant
    control at PPD30, but plots no similarity measure anywhere on the sheet. The
    evidence is the Jaccard trajectory, and it is not monotonic: similarity rises
    through PPD7 and only then falls back, so the figure has to show the shape
    rather than assert a direction. The right axes carries the counts the shape
    is computed from, which is also the manuscript's 'every stage gained more
    edges than it lost'.
    """
    rw = d["rewiring"].set_index("stage").loc[STAGES4]
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.6))
    x = np.arange(len(STAGES4))

    ax = axes[0]
    ax.plot(x, rw.jaccard.values, "-", color="#b0b0b0", lw=1.1, zorder=1)
    for xi, yi, s in zip(x, rw.jaccard.values, STAGES4):
        ax.scatter([xi], [yi], s=46, c=STAGE_COL[s], lw=0.5, edgecolors="white",
                   zorder=3)
        ax.annotate(f"{yi:.3f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.0, color=INK)
    # The descriptor goes in the empty lower-left corner rather than beside the
    # peak marker: centred under PPD7 a two-line label is wide enough to cross
    # the rising PPD1 -> PPD7 segment.
    peak = int(np.argmax(rw.jaccard.values))
    ax.text(0.02, 0.05, f"{STAGE_LAB[STAGES4[peak]]} is the most\ncontrol-like "
            "postpartum stage", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=6.0, color=STAGE_COL[STAGES4[peak]])
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LAB[s] for s in STAGES4], fontsize=7.0)
    ax.set_ylabel("Jaccard similarity\nto the control network", fontsize=6.6)
    ax.tick_params(labelsize=6.0)
    ax.margins(x=0.14)
    ax.set_ylim(0.24, 0.50)

    ax = axes[1]
    w = 0.62
    ax.bar(x, rw.gained.values / 1000.0, width=w,
           color=[STAGE_COL[s] for s in STAGES4], lw=0)
    ax.bar(x, -rw.lost.values / 1000.0, width=w,
           color=[STAGE_COL[s] for s in STAGES4], lw=0, alpha=0.42)
    ax.axhline(0.0, color=INK, lw=0.8)
    for xi, s in zip(x, STAGES4):
        ax.annotate(f"+{rw.gained[s] / 1000.0:.1f}k", (xi, rw.gained[s] / 1000.0),
                    textcoords="offset points", xytext=(0, 3), ha="center",
                    fontsize=5.8, color=INK)
        ax.annotate(f"\u2212{rw.lost[s] / 1000.0:.1f}k", (xi, -rw.lost[s] / 1000.0),
                    textcoords="offset points", xytext=(0, -9), ha="center",
                    fontsize=5.8, color="#777777")
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LAB[s] for s in STAGES4], fontsize=7.0)
    ax.set_ylabel("Interaction edges vs control\n(thousands, gained up / lost down)",
                  fontsize=6.6)
    ax.tick_params(labelsize=6.0)
    ax.margins(x=0.14)
    ax.set_ylim(-16.0, 39.0)

    fig.suptitle("Edge-set similarity to control rises through PPD7 and falls back at "
                 "PPD30; every stage gains more edges than it loses",
                 fontsize=8.0, x=0.008, ha="left")
    fig.subplots_adjust(left=0.115, right=0.99, top=0.82, bottom=0.16, wspace=0.42)
    return fig


_PANELS = [("a", panel_a, "extra_interactions_vs_depth"),
           ("b", panel_b, "extra_coverage_matrix"),
           ("c", panel_c, "SFig5AD_LR_dotplots_principal_pairs"),
           ("d", panel_d, "extra_epithelial_LR_share"),
           ("e", panel_e, "extra_rewiring_vs_control")]


def build(repo_root: str, out_dir: str, letters=None) -> dict:
    """Render every Fig S4 panel. Returns {panel letter: svg path}."""
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
        info = render_svg.verify(path + ".svg")
        if not info["text_elements"] or not info["named_font"]:
            raise RuntimeError(f"{stem}.svg exported without live text: {info}")
        plt.close(fig)
        out[letter] = path + ".svg"
    return out
