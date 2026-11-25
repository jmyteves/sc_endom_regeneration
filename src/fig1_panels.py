"""Figure 1. Gross involution and epithelial continuity.

Computable panels
  A timeline schematic (gestation | parturition | post-partum, stage ticks)
  B uterine horn length per stage: bar of mean + SEM, every animal overplotted
  C all-vs-all Tukey HSD matrix (5x5), heat = -log10(p-adj), stars in cell
  F luminal epithelium : stroma area ratio per section
  G endometrium / whole-uterus cross-section area ratio per section
  H glands per uterine cross-section (mean of two counters)

F, G and H are also written as one 7.25 x 1.6 in strip (Fig1FGH_morphometry_strip)
sized to the artboard width; the individual panels are 2.417 x 1.6 in each.
Stars in F-H are uncorrected two-tailed Student's t between all stage pairs;
Welch and Tukey p-values go to Fig1FGH_pairwise_tests.csv for the record.

Panels D and E are microscopy images. build() writes a placeholder SVG for each
and returns a 'WET-LAB: ...' string so the assembled figure has a labelled gap.

Entry point: build(repo_root, out_dir) -> {panel_label: path}
Reads repo_root/data/raw_light/horn_length_measurements.csv
              repo_root/data/raw_light/fig1/morphometry_complete.csv
"""
import itertools
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrow, Rectangle
from scipy import stats
from scipy.stats import tukey_hsd

import render_svg

# Five anatomical stages, deck order, matching the scRNA-seq design and the
# project stage palette.
STAGES = ["Control", "E16.5", "PPD1", "PPD7", "PPD30"]
STAGE_COLOR = {
    "Control": "#808080",
    "E16.5": "#cb7a29",
    "PPD1": "#A0522D",
    "PPD7": "#FA8072",
    "PPD30": "#4682B4",
}
# The horn table also carries four PPD90 rows. They are four photographs of one
# animal (M25590), not four animals, so they cannot contribute a group mean or a
# share of the pooled within-group variance. The rows stay in the CSV as a record
# of what was imaged; the filter in load_horn keeps them out of the statistics.
EXCLUDED_STAGES = ["PPD90"]
STAR_NOTE = "n.s. p>=0.05, *<.05, **<.01, ***<.001."

WETLAB = {
    "D": "WET-LAB: gross uterine horn photographs, one per stage",
    "E": "WET-LAB: EPCAM / FOXA2 immunofluorescence of the uterine wall, "
         "one per stage",
}

# Morphometry (panels F-H). Four stages: the E16.5 sections have no endometrium
# to segment, so the morphometry series starts at Control and skips E16.5.
MORPH_STAGES = ["Control", "PPD1", "PPD7", "PPD30"]
MORPH_COLOR = {"Control": "#471365", "PPD1": "#238a8d", "PPD7": "#3fbc73", "PPD30": "#bddf26"}
MORPH_PANELS = [
    # (panel, column, y label, y ticks)
    ("F", "LE_stroma_ratio", "Luminal epithelium :\nstroma area ratio", [0, 0.2, 0.4, 0.6]),
    ("G", "E_fraction", "Endometrium / uterus\narea ratio", [0, 0.2, 0.4, 0.6]),
    ("H", "gland_count_mean", "Glands per uterine\ncross-section", [0, 20, 40, 60]),
]
# Bracket rows, bottom to top. Adjacent pairs share the bottom row (a full bar
# apart), then one comparison per row ordered short to long.
BRACKET_ROWS = {"Control-PPD1": 0, "PPD7-PPD30": 0, "PPD1-PPD7": 1,
                "Control-PPD7": 2, "PPD1-PPD30": 3, "Control-PPD30": 4}
STRIP_W = 7.25
PANEL_H = 1.6


def _stars(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def load_horn(repo_root):
    """Horn-length measurements for the five figure stages, one row per animal.

    The PPD90 rows are dropped here, before any statistics, so they do not enter
    the pooled within-group variance of the ANOVA or of the Tukey HSD that reads
    the same groups.
    """
    p = os.path.join(repo_root, "data", "raw_light", "horn_length_measurements.csv")
    df = pd.read_csv(p)
    miss = set(STAGES) - set(df["stage"])
    if miss:
        raise ValueError(f"horn table is missing stage(s): {sorted(miss)}")
    df = df[~df["stage"].isin(EXCLUDED_STAGES)]
    df = df[df["stage"].isin(STAGES)].copy()
    df["stage"] = pd.Categorical(df["stage"], STAGES, ordered=True)
    return df.dropna(subset=["mean_horn_mm"])


def horn_stats(df):
    """Per-stage n / mean / SD / SEM of mean_horn_mm, in deck stage order."""
    g = df.groupby("stage", observed=True)["mean_horn_mm"]
    out = g.agg(n="count", mean="mean", sd="std").reindex(STAGES)
    out["sem"] = out["sd"] / np.sqrt(out["n"])
    return out


def tukey_matrix(df):
    """5x5 Tukey HSD p-adj matrix (DataFrame, stages on both axes)."""
    groups = [df.loc[df["stage"] == s, "mean_horn_mm"].values for s in STAGES]
    res = tukey_hsd(*groups)
    return pd.DataFrame(res.pvalue, index=STAGES, columns=STAGES)


def panel_a(out_dir):
    """Timeline schematic: gestation -> parturition -> post-partum, with ticks."""
    fig, ax = plt.subplots(figsize=(6.4, 1.55))
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.0, 1.0)
    ax.axis("off")

    # stage positions on a compressed pseudo-time axis
    pos = {"Control": -0.92, "E16.5": -0.40, "PPD1": 0.06, "PPD7": 0.34, "PPD30": 0.86}
    part = -0.12  # parturition

    ax.add_patch(Rectangle((-1.0, -0.10), (part + 1.0), 0.20, facecolor="#f0e4d4",
                           edgecolor="none", zorder=1))
    ax.add_patch(Rectangle((part, -0.10), (1.0 - part), 0.20, facecolor="#e6eef4",
                           edgecolor="none", zorder=1))
    ax.add_patch(FancyArrow(-1.0, 0.0, 1.98, 0.0, width=0.008, head_width=0.09,
                            head_length=0.05, length_includes_head=True,
                            facecolor="black", edgecolor="black", zorder=3))
    ax.plot([part, part], [-0.42, 0.46], ls=(0, (3, 2)), lw=1.0, color="black", zorder=4)
    ax.text(part, 0.52, "PARTURITION", ha="center", va="bottom", fontsize=7,
            fontweight="bold")
    ax.text((-1.0 + part) / 2, -0.30, "Gestation", ha="center", va="top", fontsize=7,
            style="italic")
    ax.text((part + 1.0) / 2, -0.30, "Post-partum", ha="center", va="top", fontsize=7,
            style="italic")

    for s in ["Control", "E16.5", "PPD1", "PPD7", "PPD30"]:
        x = pos[s]
        ax.plot([x, x], [0.0, 0.20], lw=1.0, color="black", zorder=4)
        ax.plot([x], [0.20], marker="o", ms=5.0, mfc=STAGE_COLOR[s], mec="black",
                mew=0.5, zorder=5)
        ax.text(x, 0.30, s.upper(), ha="center", va="bottom", fontsize=6.5,
                rotation=0, color="black")
    ax.text(-1.0, 0.72, "Sampling timeline", ha="left", va="bottom", fontsize=8)

    stem = os.path.join(out_dir, "Fig1A_timeline_schematic")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_b(df, out_dir):
    """Bar of mean horn length + SEM with every animal overplotted."""
    st = horn_stats(df)
    fig, ax = plt.subplots(figsize=(2.55, 2.15))
    x = np.arange(len(STAGES))
    ax.bar(x, st["mean"].values, width=0.66,
           color=[STAGE_COLOR[s] for s in STAGES], edgecolor="black", lw=0.6,
           alpha=0.92, zorder=2)
    ax.errorbar(x, st["mean"].values, yerr=st["sem"].values, fmt="none",
                ecolor="black", elinewidth=0.8, capsize=2.4, capthick=0.8, zorder=4)
    rng = np.random.default_rng(0)
    for i, s in enumerate(STAGES):
        v = df.loc[df["stage"] == s, "mean_horn_mm"].values
        ax.plot(i + rng.uniform(-0.17, 0.17, v.size), v, ls="none", marker="o",
                ms=2.6, mfc="white", mec="black", mew=0.45, alpha=0.95, zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels(STAGES, rotation=45, ha="right")
    ax.set_ylim(0, 60)
    ax.set_ylabel("Uterine horn length (mm)")
    ax.set_title("Uterine horn length", fontsize=8)
    for i, s in enumerate(STAGES):
        ax.text(i, 1.2, f"n={int(st.loc[s, 'n'])}", ha="center", va="bottom",
                fontsize=5.5, color="white", zorder=6)
    fig.tight_layout()
    stem = os.path.join(out_dir, "Fig1B_horn_length")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_c(df, out_dir):
    """All-vs-all Tukey HSD matrix, heat = -log10(p-adj), stars per cell."""
    P = tukey_matrix(df)
    n = len(STAGES)
    L = -np.log10(np.clip(P.values, 1e-16, 1.0))
    M = np.where(np.eye(n, dtype=bool), np.nan, L)
    vmax = 6.0

    fig, ax = plt.subplots(figsize=(2.95, 2.65))
    im = ax.imshow(np.ma.masked_invalid(M), cmap="magma_r", vmin=0, vmax=vmax,
                   origin="upper")
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1, facecolor="#e8e8e8",
                                       edgecolor="white", lw=0.8, zorder=2))
                ax.text(j, i, STAGES[i], ha="center", va="center", fontsize=5.6,
                        fontweight="bold", zorder=3)
            else:
                s = _stars(P.values[i, j])
                shade = "white" if L[i, j] > vmax * 0.55 else "black"
                ax.text(j, i, s, ha="center", va="center", fontsize=6.2,
                        color=shade, zorder=3)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(STAGES, rotation=45, ha="right")
    ax.set_yticklabels(STAGES)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.9)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Tukey HSD, all stage pairs", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("-log10(p-adj)", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    fig.text(0.02, 0.015, STAR_NOTE, fontsize=5.6, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    stem = os.path.join(out_dir, "Fig1C_tukey_matrix")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def _morph_stars(p):
    """Four-level star code used on the morphometry brackets."""
    if p >= 0.05:
        return "ns"
    if p >= 0.01:
        return "*"
    if p >= 0.001:
        return "**"
    if p >= 0.0001:
        return "***"
    return "****"


def load_morphometry(repo_root):
    """One row per section: stage, LE_stroma_ratio, E_fraction, gland_count_mean."""
    p = os.path.join(repo_root, "data", "raw_light", "fig1", "morphometry_complete.csv")
    if not os.path.exists(p):
        raise FileNotFoundError(f"morphometry table missing: {p}")
    df = pd.read_csv(p)
    need = {"stage"} | {c for _, c, _, _ in MORPH_PANELS}
    miss = need - set(df.columns)
    if miss:
        raise ValueError(f"morphometry table lacks column(s): {sorted(miss)}")
    df = df[df["stage"].isin(MORPH_STAGES)].copy()
    return df


def morphometry_tests(df):
    """Per-stage summary and all-pairs tests for the three morphometry metrics.

    Returns (summary, pairwise). pairwise carries Student, Welch and Tukey p for
    every pair; the panels print the Student stars.
    """
    summ, pairs = [], []
    for pan, c, _, _ in MORPH_PANELS:
        g = {s: df.loc[df["stage"] == s, c].values for s in MORPH_STAGES}
        tk = stats.tukey_hsd(*[g[s] for s in MORPH_STAGES])
        for s in MORPH_STAGES:
            summ.append(dict(panel=pan, metric=c, stage=s, n_sections=len(g[s]),
                             mean=g[s].mean(), sem=stats.sem(g[s]), sd=g[s].std(ddof=1)))
        for (i, a), (j, b) in itertools.combinations(enumerate(MORPH_STAGES), 2):
            t, p = stats.ttest_ind(g[a], g[b], equal_var=True)
            _, pw = stats.ttest_ind(g[a], g[b], equal_var=False)
            pairs.append(dict(panel=pan, metric=c, pair=f"{a}-{b}", mean_a=g[a].mean(),
                              mean_b=g[b].mean(), mean_diff=g[b].mean() - g[a].mean(),
                              t_student=t, p_student=p, p_welch=pw, p_tukey=tk.pvalue[i, j],
                              stars_student=_morph_stars(p), stars_tukey=_morph_stars(tk.pvalue[i, j]),
                              n_a=len(g[a]), n_b=len(g[b])))
    return pd.DataFrame(summ), pd.DataFrame(pairs)


def _draw_morph(ax, df, pairs, pan, c, ylabel, yticks):
    """Bar of mean + SEM, white points per section, all-vs-all star brackets."""
    g = {s: df.loc[df["stage"] == s, c].values for s in MORPH_STAGES}
    x = np.arange(len(MORPH_STAGES))
    means = [g[s].mean() for s in MORPH_STAGES]
    sems = [stats.sem(g[s]) for s in MORPH_STAGES]
    ax.bar(x, means, width=0.62, color=[MORPH_COLOR[s] for s in MORPH_STAGES],
           edgecolor="none", zorder=1)
    ax.errorbar(x, means, yerr=sems, fmt="none", ecolor="black", elinewidth=0.8,
                capsize=2, capthick=0.8, zorder=3)
    rng = np.random.default_rng(1)
    for i, s in enumerate(MORPH_STAGES):
        ax.scatter(x[i] + rng.uniform(-0.18, 0.18, len(g[s])), g[s], s=9,
                   facecolor="white", edgecolor="black", linewidth=0.5, zorder=4)
    ymax = max(v.max() for v in g.values())
    step = 0.12 * ymax
    base = ymax + 0.6 * step
    sp = pairs[pairs["panel"] == pan].set_index("pair")
    for pair, row in BRACKET_ROWS.items():
        a, b = pair.split("-")
        xa, xb = MORPH_STAGES.index(a), MORPH_STAGES.index(b)
        y = base + row * step
        ax.plot([xa, xa, xb, xb], [y - 0.25 * step, y, y, y - 0.25 * step],
                color="black", lw=0.6, clip_on=False)
        lab = sp.loc[pair, "stars_student"]
        ax.text((xa + xb) / 2, y + 0.08 * step, lab, ha="center",
                va="bottom" if lab == "ns" else "center", fontsize=5 if lab == "ns" else 6)
    ax.set_xticks(x)
    ax.set_xticklabels(MORPH_STAGES, fontsize=6)
    ax.set_yticks(yticks)
    ax.tick_params(axis="y", labelsize=6)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_ylim(0, base + 5 * step + 0.4 * step)
    ax.set_xlim(-0.6, 3.6)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def panels_fgh(df, out_dir):
    """Morphometry strip (7.25 x 1.6 in) plus the three panels at slot size."""
    summ, pairs = morphometry_tests(df)
    render_svg.arial_mode(base=7, secondary=6, tick=6)
    pw = STRIP_W / 3
    lg = 0.60 / pw   # left gutter per panel: two-line 7 pt y labels need ~0.55 in
    out = {}

    fig, axes = plt.subplots(1, 3, figsize=(STRIP_W, PANEL_H))
    for ax, (pan, c, yl, yt) in zip(axes, MORPH_PANELS):
        _draw_morph(ax, df, pairs, pan, c, yl, yt)
    fig.subplots_adjust(left=lg / 3, right=1 - 0.02 / 3, bottom=0.17, top=0.98,
                        wspace=lg / (1 - lg))
    stem = os.path.join(out_dir, "Fig1FGH_morphometry_strip")
    out["FGH"] = render_svg.save_fixed(fig, stem)
    plt.close(fig)

    for pan, c, yl, yt in MORPH_PANELS:
        fig, ax = plt.subplots(figsize=(pw, PANEL_H))
        _draw_morph(ax, df, pairs, pan, c, yl, yt)
        fig.subplots_adjust(left=lg, right=0.98, bottom=0.17, top=0.98)
        stem = os.path.join(out_dir, f"Fig1{pan}_{c}")
        out[pan] = render_svg.save_fixed(fig, stem)
        plt.close(fig)

    summ.to_csv(os.path.join(out_dir, "Fig1FGH_summary.csv"), index=False)
    pairs.to_csv(os.path.join(out_dir, "Fig1FGH_pairwise_tests.csv"), index=False)
    return out, summ, pairs


def _wetlab_slug(label, text, out_dir):
    """A placeholder SVG stating what wet-lab asset belongs in this panel."""
    fig, ax = plt.subplots(figsize=(2.5, 1.5))
    ax.axis("off")
    ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
                           facecolor="none", edgecolor="#999999", lw=0.8,
                           ls=(0, (4, 3))))
    ax.text(0.5, 0.72, f"Panel {label}", transform=ax.transAxes, ha="center",
            va="center", fontsize=9, fontweight="bold", color="#555555")
    body = text.split(": ", 1)[1]
    ax.text(0.5, 0.44, "\n".join(_wrap(body, 34)), transform=ax.transAxes,
            ha="center", va="center", fontsize=6, color="#555555")
    ax.text(0.5, 0.10, "wet-lab image asset - not computed", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.5, style="italic", color="#888888")
    stem = os.path.join(out_dir, f"Fig1{label}_WETLAB_placeholder")
    render_svg.save_svg(fig, stem, also_pdf=False)
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


def build(repo_root: str, out_dir: str) -> dict:
    """Render every computable Fig1 panel; write a placeholder for the wet-lab ones."""
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    df = load_horn(repo_root)
    paths = {
        "A": panel_a(out_dir),
        "B": panel_b(df, out_dir),
        "C": panel_c(df, out_dir),
    }
    for lab, text in WETLAB.items():
        _wetlab_slug(lab, text, out_dir)
        paths[lab] = text
    morph, _, _ = panels_fgh(load_morphometry(repo_root), out_dir)
    paths.update(morph)
    return paths
