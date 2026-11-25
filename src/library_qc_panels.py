"""Library quality-control panels (fig. S2 A, B, D) and two extra flow panels.

    load_inputs(repo_root) -> dict of source tables (data/raw_light/sfig1/)
    panel_i  cells passing QC per library, log axis, CTRL_3 marked excluded   fig. S2 A
    panel_k  count-feature relation on library medians with the recorded fit  fig. S2 B
    panel_j  per-stage QC medians: genes, UMIs, mitochondrial percent          fig. S2 D
    panel_g  flow statistics on disk for the per-stage tubes: gate ladder,
             7-AAD/PI detector diagnostic, marker positivity, singlet yield    extra
    panel_h  doublet handling per library                                      extra

The flow tables describe the per-stage tubes, a different experiment from the
30/45/60 min digestion series of fig. S1, which is why panel_g is not a fig. S1 panel.

Two cell denominators appear in these panels and both are correct:
  698,595 over 14 libraries, CTRL_3 excluded: what panels i, j, k count
  698,631 over 15 libraries: the atlas object, with CTRL_3's 36 cells
"""
import os

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

# Library order: stage order, then the within-stage index. CTRL_3 sits last in
# its stage so the excluded marker reads as an appendix to CTRL, not a gap.
LIBRARY_ORDER = ["CTRL_1", "CTRL_2", "CTRL_3",
                 "E16_5_1", "E16_5_2", "E16_5_3",
                 "PPD1_1", "PPD1_2", "PPD1_3",
                 "PPD7_1", "PPD7_2", "PPD7_3",
                 "PPD30_1", "PPD30_2", "PPD30_3"]
EXCLUDED_LIBRARY = "CTRL_3"

# Panels a-f, with the reason each is unbuildable printed into its slug.
NOT_COMPUTABLE = {
    "a": "gating density plots, 30 min digestion column (7-AAD, EpCAM, PDGFRa, "
         "CD45, CD31, unstained SSC-A): raw FCS only, no gate export",
    "b": "gating density plots, 45 min digestion column: raw FCS only, "
         "no gate export",
    "c": "gating density plots, 60 min digestion column: raw FCS only, "
         "no gate export",
    "d": "percent of live cells positive per marker by digestion time: needs a "
         "live gate, and 7-AAD gates were never exported",
    "e": "viability as percent of singlets by digestion time (deck 81.0 / 56.4 "
         "/ 60.0): no gate statistics on disk, and re-gating invents the number",
    "f": "total live cells per mg by digestion time (deck 5.9 / 12.7 / 3.0): no "
         "FCS keyword or table records a tissue mass",
}

# The deck reports one figure per stage for the QC medians. Those are the MEAN of
# the per-library medians: the per-cell distributions are not on disk, so a
# pooled median cannot be formed. Taking the median of the library medians
# instead disagrees with the deck at E16.5 (3011 against 2384), so the mean is
# the aggregation the deck used.
STAGE_AGG = "mean of per-library medians"

# 200 genes and 20% mitochondrial reads are absolute cutoffs and can be drawn.
# The 0.5th-percentile gene floor and 99.5th-percentile UMI ceiling are
# per-library percentiles, so they land at a different value in every library
# and no single line represents them.
MIN_GENES = 200
MAX_PCT_MITO = 20


def _p(repo_root, name):
    return os.path.join(repo_root, "data", "raw_light", "sfig1", name)


def load_inputs(repo_root):
    """The five SFig1 source tables, keyed by the panel that consumes them."""
    want = {
        "qc_cells": "S1e_cells_passing_qc_source.csv",
        "qc_dist": "S1f_qc_distributions_source.csv",
        "count_feature": "S1g_count_feature_filter_source.csv",
        "doublet": "S1h_doublet_summary_source.csv",
        "gate_ladder": "FL_a_gating_strategy_source.csv",
        "pi_spillover": "FL_b_PI_channel_is_spillover_source.csv",
        "markers_by_stage": "FL_c_compartment_markers_by_stage_source.csv",
        "per_tube_qc": "FL_d_per_tube_qc_source.csv",
    }
    out = {}
    for key, fname in want.items():
        path = _p(repo_root, fname)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"SFig1 input missing: {path} (copy it from "
                f"sc_endom_EDITABLE_SVG/SFigs_latest/tables/{fname})")
        out[key] = pd.read_csv(path)
    n = len(out["qc_cells"])
    if n != 15:
        raise ValueError(f"S1e should carry 15 libraries, got {n}")
    for key in ("qc_dist", "count_feature", "doublet"):
        if len(out[key]) != 14:
            raise ValueError(f"{key} should carry the 14 analysed libraries, "
                             f"got {len(out[key])}")
    return out


def stage_medians(qc_dist):
    """Per-stage QC figures as the mean of the per-library medians."""
    cols = ["median_detected_genes", "median_total_umis", "median_pct_mito"]
    g = qc_dist.groupby("stage", observed=True)[cols].mean()
    return g.reindex(STAGES)


def stage_totals(qc_cells):
    """Cells per stage over the analysed libraries, and the 14-library total."""
    inc = qc_cells[qc_cells["included"]]
    per = inc.groupby("stage", observed=True)["cells_passing_qc"].sum()
    per = per.reindex(STAGES)
    return per, int(per.sum()), int(len(inc))


def count_feature_fit(cf):
    """The across-library fit as recorded, plus a refit that has to agree.

    The filter was applied as a per-library fit over per-cell points. Neither
    the per-cell cloud nor its fit is on disk, so the panel shows the
    library-median relationship and the fit the table records. Refitting the 14
    library medians here is a check on the recorded coefficients, not a
    replacement for them.
    """
    rec = {"slope": cf["fit_slope"].iloc[0], "intercept": cf["fit_intercept"].iloc[0],
           "r": cf["fit_r"].iloc[0], "r2": cf["fit_r2"].iloc[0],
           "band": abs(cf["per_cell_boundary_residual_units"].iloc[0])}
    if cf["fit_slope"].nunique() != 1:
        raise ValueError("S1g carries more than one fit slope")
    slope, intercept = np.polyfit(cf["log_med_counts"], cf["log_med_genes"], 1)
    r = np.corrcoef(cf["log_med_counts"], cf["log_med_genes"])[0, 1]
    refit = {"slope": slope, "intercept": intercept, "r": r, "r2": r ** 2}
    for k in ("slope", "intercept", "r"):
        if abs(refit[k] - rec[k]) > 1e-6:
            raise ValueError(f"refit {k} {refit[k]:.6f} disagrees with the "
                             f"recorded {rec[k]:.6f} in S1g")
    return rec


def panel_g(tab, out_dir):
    """Flow statistics on disk: gate ladder, detector diagnostic, positivity, yield.

    Session-A per-stage tubes. The digestion series has no gate export, so this
    panel carries the flow evidence that exists rather than the deck's columns.
    """
    ladder = tab["gate_ladder"]
    spill = tab["pi_spillover"]
    mk = tab["markers_by_stage"]
    tubes = tab["per_tube_qc"]

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6))
    ax = axes[0, 0]
    names = ["Events", "Cells\n(FSC-A Otsu)", "Singlets\n(FSC-H/A)"]
    vals = [ladder["n_in"].iloc[0], ladder["n_in"].iloc[1], ladder["n_in"].iloc[2]]
    ax.barh(range(3), vals, color="#748190", edgecolor="black", lw=0.6, height=0.6)
    for i, (v, lab) in enumerate(zip(vals, names)):
        ax.text(v * 1.02, i, f"{int(v):,}", va="center", fontsize=6)
    ax.set_yticks(range(3))
    ax.set_yticklabels(names, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlim(0, max(vals) * 1.28)
    ax.set_xlabel("Events")
    ax.set_title("Gate ladder, Control ALLCOLORS (session A)", fontsize=7.5)
    ax.text(0.98, 0.30, f"{ladder['pct_retained'].iloc[0]:.2f}% of events\n"
            f"{ladder['pct_retained'].iloc[1]:.2f}% of cells",
            transform=ax.transAxes, ha="right", va="center", fontsize=5.8)
    ax.text(0.98, 0.06, "live/dead gate not applied:\nPI and PDGFRa share Y610",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6,
            style="italic", color="#a03020")

    # FL4 and FL6 track FL5 at the spillover coefficient the instrument recorded,
    # which is why no viability gate is defensible in this panel.
    ax = axes[0, 1]
    for det, mfc in (("FL4-A", "white"), ("FL6-A", "#367f80")):
        s = spill[spill["detector"] == det]
        ax.plot(s["recorded_spillover_from_FL5"], s["fitted_slope_vs_FL5"],
                ls="none", marker="o", ms=4.2, mfc=mfc, mec="black", mew=0.5,
                label=f"{det} ({s['detector_name'].iloc[0]})")
    lo, hi = 0.28, 0.46
    ax.plot([lo, hi], [lo, hi], ls=(0, (3, 2)), lw=0.8, color="#888888", zorder=1)
    ax.set_xlabel("Recorded spillover from FL5")
    ax.set_ylabel("Fitted slope against FL5")
    ax.set_title("7-AAD / PI channels are FL5 spillover", fontsize=7.5)
    ax.legend(fontsize=5.4, loc="upper left")
    ax.text(0.97, 0.05, f"R2 {spill['r_squared'].min():.2f} to "
            f"{spill['r_squared'].max():.3f}, {len(spill)} tube x detector",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.6)

    ax = axes[1, 0]
    x = np.arange(len(STAGES))
    m = mk.set_index("stage").reindex(STAGES)
    ax.bar(x - 0.19, m["pct_epcam_pos"], width=0.34, color="#748190",
           edgecolor="black", lw=0.5, label="EpCAM+")
    ax.bar(x + 0.19, m["pct_cd45_pos"], width=0.34, color="#c58a4c",
           edgecolor="black", lw=0.5, label="CD45+")
    for i, s in enumerate(STAGES):
        ax.text(i, 2, "n.d.", ha="center", va="bottom", fontsize=5.0,
                color="#a03020", rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES], rotation=45, ha="right")
    ax.set_ylabel("% of singlets above unstained p99")
    ax.set_ylim(0, 105)
    ax.set_title("Marker positivity per stage", fontsize=7.5)
    ax.legend(fontsize=5.6, loc="upper center", ncol=2)
    ax.text(0.02, 0.92, "n.d.: PDGFRa confounded on Y610, CD31 absent from panel",
            transform=ax.transAxes, fontsize=5.4, color="#a03020")

    ax = axes[1, 1]
    t = tubes.sort_values(["session", "tube_name"])
    col = ["#367f80" if s.startswith("A") else "#e58677" for s in t["session"]]
    ax.bar(range(len(t)), t["pct_singlets_of_cells"], color=col,
           edgecolor="black", lw=0.4)
    ax.set_xticks(range(len(t)))
    ax.set_xticklabels(t["tube_name"], rotation=90, fontsize=4.6)
    ax.set_ylabel("% singlets of cells")
    ax.set_ylim(0, 105)
    ax.set_title("Singlet yield per tube", fontsize=7.5)
    ax.legend(handles=[Line2D([], [], marker="s", ls="none", mfc="#367f80",
                              mec="black", ms=4, label="session A 17-Feb-2026"),
                       Line2D([], [], marker="s", ls="none", mfc="#e58677",
                              mec="black", ms=4, label="session B 20-Jul-2026")],
              fontsize=5.2, loc="lower left")
    fig.suptitle("Flow cytometry statistics available on disk "
                 "(per-stage tubes, automated gates)", fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    stem = os.path.join(out_dir, "extra_flow_gate_statistics")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_h(doublet, out_dir):
    """Doublet handling. The per-library rate is zero by construction, not measured."""
    d = doublet.set_index("replicate").reindex(
        [r for r in LIBRARY_ORDER if r in set(doublet["replicate"])])
    delivered = doublet[
        "delivered_SFig12_atlas_residual_score_gt_0.5_pct"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.5),
                             gridspec_kw={"width_ratios": [3.2, 1.0]})
    ax = axes[0]
    ax.bar(range(len(d)), d["pct_doublet_calls_among_retained"], width=0.68,
           color=[STAGE_COLOR[s] for s in d["stage"]], edgecolor="black", lw=0.5)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([r.replace("_", "-") for r in d.index], rotation=90,
                       fontsize=5.4)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("% doublet calls among retained cells")
    ax.set_title("Doublet calls in the retained cells", fontsize=8)
    ax.text(0.5, 0.55, "0.00% in all 14 libraries\n"
            "predicted doublets were removed before this table was written, so "
            "the retained set\ncontains singlets only: this is a post-filter "
            "call rate, not a measured doublet rate",
            transform=ax.transAxes, ha="center", va="center", fontsize=5.8,
            color="#444444")
    ax.text(0.02, 0.94, f"scDblFinder {doublet['scDblFinder_version'].iloc[0]}, "
            f"run per {doublet['run_unit'].iloc[0]}", transform=ax.transAxes,
            fontsize=5.6)

    ax = axes[1]
    ax.bar([0], [delivered], width=0.5, color="#748190", edgecolor="black", lw=0.6)
    ax.text(0, delivered + 0.02, f"{delivered:.2f}%", ha="center", va="bottom",
            fontsize=6.5)
    ax.set_xticks([0])
    ax.set_xticklabels(["residual score\n> 0.5"], fontsize=5.8)
    ax.set_xlim(-0.6, 0.6)
    ax.set_ylim(0, 0.5)
    ax.set_ylabel("% of atlas cells")
    ax.set_title("Delivered SFig12 figure", fontsize=7.5)
    ax.text(0.5, 0.55, "a score-based residual\nover the whole atlas,\n"
            "a different quantity\nfrom the call rate at left",
            transform=ax.transAxes, ha="center", va="center", fontsize=5.4,
            style="italic", color="#444444")
    fig.tight_layout()
    stem = os.path.join(out_dir, "extra_doublet_summary")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_i(qc_cells, out_dir):
    """Cells passing QC per library, log axis, CTRL_3 marked as excluded."""
    d = qc_cells.set_index("replicate").reindex(LIBRARY_ORDER)
    per_stage, total, n_lib = stage_totals(qc_cells)

    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for i, (rep, row) in enumerate(d.iterrows()):
        c = STAGE_COLOR[row["stage"]]
        keep = bool(row["included"])
        ax.plot([i, i], [1, row["cells_passing_qc"]], lw=1.0,
                color=c if keep else "#b03020", zorder=2)
        ax.plot([i], [row["cells_passing_qc"]], marker="o", ms=5.2,
                mfc=c if keep else "white", mec="black" if keep else "#b03020",
                mew=0.6 if keep else 1.0, zorder=3)
        ax.text(i, row["cells_passing_qc"] * 1.35, f"{int(row['cells_passing_qc']):,}",
                ha="center", va="bottom", fontsize=5.2, rotation=90)
    ax.set_yscale("log")
    ax.set_ylim(20, 6e5)
    ax.set_xlim(-0.8, len(d) - 0.2)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([r.replace("_", "-") for r in d.index], rotation=90,
                       fontsize=6)
    ax.set_ylabel("Cells passing QC")
    ax.set_title("Cells passing QC per library", fontsize=8.5)

    j = list(d.index).index(EXCLUDED_LIBRARY)
    ax.annotate("CTRL-3 excluded", xy=(j, d["cells_passing_qc"].iloc[j]),
                xytext=(j + 1.1, 250), fontsize=6.2, color="#b03020",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="#b03020"))

    # Stage bands carry the per-stage totals, so the reader gets the denominator
    # of every downstream per-stage statistic without leaving the panel.
    start = 0
    for s in STAGES:
        n = int((d["stage"] == s).sum())
        mid = start + (n - 1) / 2
        ax.text(mid, 4.2e5, f"{STAGE_LABEL[s]}\n{round(per_stage[s] / 1000):,}k",
                ha="center", va="center", fontsize=6.2, color=STAGE_COLOR[s],
                fontweight="bold")
        if start:
            ax.axvline(start - 0.5, lw=0.5, color="#cccccc", zorder=1)
        start += n
    ax.text(0.99, 0.02, f"{total:,} cells / {n_lib} libraries "
            f"(CTRL_3 excluded; the atlas object holds 698,631 over 15)",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=5.8)
    fig.tight_layout()
    stem = os.path.join(out_dir, "SFig2A_cells_passing_qc")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_j(qc_dist, out_dir):
    """Per-stage QC medians with every library overplotted and the drawable cutoffs."""
    st = stage_medians(qc_dist)
    specs = [("median_detected_genes", "Median genes per cell", True, MIN_GENES,
              "200-gene minimum"),
             ("median_total_umis", "Median UMIs per cell", True, None, None),
             ("median_pct_mito", "Median mitochondrial %", False, MAX_PCT_MITO,
              "20% maximum")]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    x = np.arange(len(STAGES))
    for ax, (col, ylab, logy, cut, cutlab) in zip(axes, specs):
        ax.bar(x, st[col].values, width=0.64,
               color=[STAGE_COLOR[s] for s in STAGES], edgecolor="black", lw=0.5,
               alpha=0.92, zorder=2)
        for i, s in enumerate(STAGES):
            v = qc_dist.loc[qc_dist["stage"] == s, col].values
            ax.plot(np.full(v.size, i), v, ls="none", marker="o", ms=2.8,
                    mfc="white", mec="black", mew=0.45, zorder=4)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(100, 1.2e4)
        if cut is not None:
            ax.axhline(cut, ls=(0, (3, 2)), lw=0.8, color="#b03020", zorder=3)
            ax.text(len(STAGES) - 0.5, cut, cutlab, ha="right", va="bottom",
                    fontsize=5.2, color="#b03020")
        for i, s in enumerate(STAGES):
            v = st.loc[s, col]
            txt = f"{v:,.0f}" if col != "median_pct_mito" else f"{v:.1f}"
            ax.text(i, v, txt, ha="center", va="bottom", fontsize=5.4,
                    rotation=90, zorder=5)
        ax.set_xticks(x)
        ax.set_xticklabels([STAGE_LABEL[s] for s in STAGES], rotation=90)
        ax.set_ylabel(ylab, fontsize=6.5)
    axes[2].set_ylim(0, 24)
    fig.suptitle(f"Library QC by stage ({STAGE_AGG}, 14 libraries)", fontsize=8.5)
    fig.text(0.5, 0.005, "The 0.5th-percentile gene floor and 99.5th-percentile "
             "UMI ceiling are per-library percentiles, so neither is drawable as "
             "one line.", ha="center", va="bottom", fontsize=5.4)
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    stem = os.path.join(out_dir, "SFig2D_qc_distributions")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_k(cf, out_dir):
    """Count-feature relation on library medians, log-log, with fit and band."""
    fit = count_feature_fit(cf)
    fig, ax = plt.subplots(figsize=(3.9, 3.4))
    xs = np.linspace(cf["log_med_counts"].min() - 0.25,
                     cf["log_med_counts"].max() + 0.25, 50)
    ys = fit["intercept"] + fit["slope"] * xs
    ax.fill_between(xs, ys - fit["band"], ys + fit["band"], color="#cccccc",
                    alpha=0.45, lw=0, zorder=1)
    ax.plot(xs, ys, lw=1.0, color="black", zorder=2)
    for s in STAGES:
        d = cf[cf["stage"] == s]
        ax.plot(d["log_med_counts"], d["log_med_genes"], ls="none", marker="o",
                ms=5.0, mfc=STAGE_COLOR[s], mec="black", mew=0.5,
                label=STAGE_LABEL[s], zorder=3)
    worst = cf.loc[cf["residual_from_across_library_fit"].idxmin()]
    ax.annotate(f"{worst['replicate'].replace('_', '-')}\n"
                f"residual {worst['residual_from_across_library_fit']:.2f}",
                xy=(worst["log_med_counts"], worst["log_med_genes"]),
                xytext=(worst["log_med_counts"] + 0.15,
                        worst["log_med_genes"] - 0.55),
                fontsize=5.8, ha="left",
                arrowprops=dict(arrowstyle="-", lw=0.7, color="black"))
    ax.set_xlabel("log median UMIs per cell")
    ax.set_ylabel("log median genes per cell")
    ax.set_title("Count-feature relation, library medians", fontsize=8)
    ax.legend(fontsize=5.6, loc="upper left", ncol=2)
    ax.text(0.98, 0.06, f"slope {fit['slope']:.3f}, r {fit['r']:.3f}, "
            f"R2 {fit['r2']:.3f}\nband +/-{fit['band']:.1f} residual units, "
            f"n = {len(cf)} libraries", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=5.8)
    fig.tight_layout()
    stem = os.path.join(out_dir, "SFig2B_count_feature_fit")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def _wrap(s, width):
    out, line = [], ""
    for w in s.split():
        if len(line) + len(w) + 1 > width:
            out.append(line); line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        out.append(line)
    return out


def _slug(label, text, out_dir):
    """A placeholder SVG naming what belongs in the panel and why it is absent."""
    fig, ax = plt.subplots(figsize=(2.6, 1.7))
    ax.axis("off")
    ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
                           facecolor="none", edgecolor="#999999", lw=0.8,
                           ls=(0, (4, 3))))
    ax.text(0.5, 0.84, f"Panel {label}", transform=ax.transAxes, ha="center",
            va="center", fontsize=9, fontweight="bold", color="#555555")
    ax.text(0.5, 0.46, "\n".join(_wrap(text, 36)), transform=ax.transAxes,
            ha="center", va="center", fontsize=5.6, color="#555555")
    ax.text(0.5, 0.07, "not computable from any table on disk",
            transform=ax.transAxes, ha="center", va="center", fontsize=5.4,
            style="italic", color="#888888")
    stem = os.path.join(out_dir, f"SFig1{label}_NOTCOMPUTABLE_placeholder")
    render_svg.save_svg(fig, stem, also_pdf=False)
    plt.close(fig)
    return stem + ".svg"


def build(repo_root: str, out_dir: str) -> dict:
    """Render the computable SFig1 panels; slug the digestion-series ones."""
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    tab = load_inputs(repo_root)

    paths = {}
    for lab, reason in NOT_COMPUTABLE.items():
        _slug(lab, reason, out_dir)
        paths[lab] = f"NOT COMPUTABLE: {reason}"
    paths["g"] = panel_g(tab, out_dir)
    paths["h"] = panel_h(tab["doublet"], out_dir)
    paths["i"] = panel_i(tab["qc_cells"], out_dir)
    paths["j"] = panel_j(tab["qc_dist"], out_dir)
    paths["k"] = panel_k(tab["count_feature"], out_dir)
    return paths
