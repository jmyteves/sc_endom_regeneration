"""Compartment coordination and anchor invariance (extra panels, former fig. S5).

    build(repo_root, out_dir) -> {letter: svg_path}

  a  mean correlation of each compartment's stage response with the other three,
     per stage, against the permutation null band; four compartment lines plus the
     within-stage mean of all six pairs in grey.
  b  between-stage similarity of the response direction, 4 x 4.
  c  per-stage cross-compartment correlation matrices, four 4 x 4 matrices on one
     colour scale.
  d  layer retention recomputed under three anchor choices (CTRL, E16.5, PPD1), one
     sub-axes per layer, with the anchor-agreement summary underneath.

CTRL is absent from a-c: the coordinated quantity is each compartment's response
direction relative to control. Retention in d is scaled within each anchor column
(anchor stage at 0, most divergent stage at 1), which is why the agreement table reports
CTRL_range = 1.00 for all six layers. Every plotted number comes from the staged source
tables unchanged. Figures are saved at true physical size, so a fontsize of N is N pt on
the page; save_svg crops to the drawn bbox without scaling.
"""
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import render_svg

MM = 1 / 25.4

STAGES = ["E16_5", "PPD1", "PPD7", "PPD30"]
STAGES5 = ["CTRL", "E16_5", "PPD1", "PPD7", "PPD30"]
STAGE_LAB = {"CTRL": "Control", "E16_5": "E16.5", "PPD1": "PPD1",
             "PPD7": "PPD7", "PPD30": "PPD30"}
STAGE_COL = {"CTRL": "#808080", "E16_5": "#cb7a29", "PPD1": "#A0522D",
             "PPD7": "#FA8072", "PPD30": "#4682B4"}

COMPS = ["Epithelial", "Stromal", "Endothelial", "Immune"]
CPAL = {"Epithelial": "#748190", "Stromal": "#367f80",
        "Endothelial": "#e58677", "Immune": "#c58a4c"}

LAYERS = ["Anatomy", "Identity", "Composition", "State", "Compartment", "Signaling"]
ANCHORS = ["CTRL", "E16_5", "PPD1"]

# Agreement between two anchor choices, one column per anchor pair. The first is the
# column the deck prints as its anchor-sensitivity readout.
AGREE = [("r_CTRL_vs_E16_5", "CTRL vs E16.5"),
         ("r_CTRL_vs_PPD1", "CTRL vs PPD1"),
         ("r_PPD1_vs_E16_5", "PPD1 vs E16.5")]
AGREE_COL = ["#3d4550", "#8b95a1", "#c9ced5"]
INVARIANCE_THRESHOLD = 0.75

GREY = "#555555"
BAND = "#bfc4c9"

_TABLES = {
    "percomp": "S5f_coordination_matrices_per_compartment_source.csv",   # panel a
    "within": "S5f_coordination_matrices_within_stage_source.csv",       # panel a reference
    "between": "S5f_coordination_matrices_between_stage_source.csv",     # panel b
    "pairwise": "S5f_coordination_matrices_pairwise_source.csv",         # panel c
    "anchor": "S5g_layer_retention_by_anchor_source.csv",                # panel d
    "agree": "S5g_layer_retention_anchor_agreement.csv",                 # panel d summary
}

# Two further tables on the Drive are byte-for-byte duplicates of ones read above and are
# not staged: S5f_coordination_pairwise_reconstructed.csv repeats the pairwise source and
# S5g_layer_retention_anchor_sensitivity_source.csv repeats the by-anchor source.
# S5f_per_compartment_coordination_with_null.csv likewise repeats the per-compartment source.


def load_inputs(repo_root):
    """Read the six staged coordination tables and check their shapes."""
    d = os.path.join(repo_root, "data", "raw_light", "sfig5")
    out = {}
    for key, fname in _TABLES.items():
        path = os.path.join(d, fname)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                "SFig5 input missing: " + path + "\nStage it from "
                "sc_endom_EDITABLE_SVG/SFigs_latest/tables/" + fname)
        out[key] = pd.read_csv(path)
    shapes = {"percomp": 16, "within": 4, "between": 4, "pairwise": 24,
              "anchor": 15, "agree": 6}
    for key, n in shapes.items():
        if len(out[key]) != n:
            raise ValueError(f"{_TABLES[key]}: expected {n} rows, got {len(out[key])}")
    if list(out["agree"].layer) != LAYERS:
        raise ValueError("anchor-agreement layer order drifted: "
                         + ", ".join(out["agree"].layer))
    return out


def _matrix(pairwise, stage):
    """One stage's 6 pairwise r values -> a symmetric 4 x 4 frame, diagonal NaN.

    The diagonal is left empty rather than set to 1: a compartment's correlation with
    itself is 1 by construction and would dominate the colour scale.
    """
    m = pd.DataFrame(np.nan, index=COMPS, columns=COMPS, dtype=float)
    sub = pairwise[pairwise.stage == stage]
    if len(sub) != 6:
        raise ValueError(f"{stage}: expected 6 compartment pairs, got {len(sub)}")
    for _, r in sub.iterrows():
        m.loc[r.compartment_a, r.compartment_b] = r.pearson_r
        m.loc[r.compartment_b, r.compartment_a] = r.pearson_r
    return m


def _annotate_matrix(ax, m, vmin, vmax, cmap="YlGnBu", fs=6.0, fmt="{:.2f}"):
    """Draw a labelled matrix; light text on the dark end so every cell stays legible."""
    im = ax.imshow(m.values, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(range(len(m.columns)))
    ax.set_yticks(range(len(m.index)))
    span = vmax - vmin
    for i in range(len(m.index)):
        for j in range(len(m.columns)):
            v = m.values[i, j]
            if np.isnan(v):
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none",
                                       edgecolor="none"))
                continue
            col = "white" if (v - vmin) / span > 0.62 else "#1a1a1a"
            ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=fs, color=col)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def panel_a(d):
    """Mean r to the other three compartments, per stage, against the null band."""
    pc, ws = d["percomp"], d["within"]
    fig = plt.figure(figsize=(112 * MM, 74 * MM))
    ax = fig.add_axes([0.11, 0.13, 0.60, 0.80])
    x = np.arange(len(STAGES))

    # The permutation null is centred on zero; its width comes from the SD implied by the
    # tabulated z, so the band is the same object the z column was computed against.
    sd = pc.implied_null_sd.max()
    ax.axhspan(-1.96 * sd, 1.96 * sd, color=BAND, zorder=0)
    ax.axhline(0.0, color=GREY, lw=0.5, ls=(0, (3, 3)), zorder=1)

    ax.plot(x, [ws.set_index("stage").within_stage_r[s] for s in STAGES], color=GREY,
            lw=1.6, ls=(0, (4, 2)), marker="D", ms=3.0, zorder=2,
            label="all six pairs")
    for c in COMPS:
        sub = pc[pc.compartment == c].set_index("stage")
        ax.plot(x, [sub.mean_r_to_others[s] for s in STAGES], color=CPAL[c], lw=1.6,
                marker="o", ms=3.4, zorder=3, label=c)

    zmin = pc.loc[pc.z.idxmin()]
    ax.annotate(f"weakest z = {zmin.z:.1f}\n({STAGE_LAB[zmin.stage]} {zmin.compartment})",
                xy=(STAGES.index(zmin.stage), zmin.mean_r_to_others),
                xytext=(STAGES.index(zmin.stage) - 0.42, zmin.mean_r_to_others - 0.16),
                fontsize=5.6, color=GREY, va="top",
                arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY,
                                shrinkA=0, shrinkB=2))
    ax.text(-0.42, 1.96 * sd + 0.012, f"permutation null, mean $\\pm$ 1.96 SD "
            f"($\\pm${1.96 * sd:.3f})", fontsize=5.6, color=GREY, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels([STAGE_LAB[s] for s in STAGES])
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    ax.set_ylim(-0.06, 0.86)
    ax.set_ylabel("mean Pearson $r$ to the other\ncompartments' stage response")
    ax.set_title("Every compartment moves with the rest at every stage", fontsize=7.5,
                 loc="left", pad=4)
    ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=6, handlelength=1.6,
              borderaxespad=0)
    return fig


def panel_b(d):
    """Between-stage similarity of the response direction."""
    bt = d["between"].set_index("stage").loc[STAGES, STAGES]
    fig = plt.figure(figsize=(84 * MM, 76 * MM))
    ax = fig.add_axes([0.20, 0.16, 0.62, 0.68])
    im = _annotate_matrix(ax, bt, vmin=0.0, vmax=1.0, fs=6.4)
    ax.set_xticklabels([STAGE_LAB[s] for s in STAGES])
    ax.set_yticklabels([STAGE_LAB[s] for s in STAGES])
    for i, s in enumerate(STAGES):
        ax.get_xticklabels()[i].set_color(STAGE_COL[s])
        ax.get_yticklabels()[i].set_color(STAGE_COL[s])
    ax.set_title("Response direction shared between stages", fontsize=7.5, loc="left",
                 pad=5)
    cax = fig.add_axes([0.855, 0.16, 0.028, 0.68])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("similarity ($r$)", fontsize=6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, labelsize=5.6)
    return fig


def panel_c(d):
    """Four per-stage cross-compartment matrices on one shared scale."""
    pw = d["pairwise"]
    mats = {s: _matrix(pw, s) for s in STAGES}
    vmax = 0.8
    fig = plt.figure(figsize=(184 * MM, 58 * MM))
    left, w, gap = 0.055, 0.195, 0.026
    ims = None
    for k, s in enumerate(STAGES):
        ax = fig.add_axes([left + k * (w + gap), 0.20, w, 0.60])
        ims = _annotate_matrix(ax, mats[s], vmin=0.0, vmax=vmax, fs=5.8)
        ax.set_xticklabels([c[:4] for c in COMPS], rotation=45, ha="right")
        if k == 0:
            ax.set_yticklabels(COMPS)
        else:
            ax.set_yticklabels([])
        ax.set_title(STAGE_LAB[s], fontsize=7, color=STAGE_COL[s], pad=3)
    cax = fig.add_axes([left + 4 * (w + gap) - 0.004, 0.20, 0.012, 0.60])
    cb = fig.colorbar(ims, cax=cax)
    cb.set_label("Pearson $r$", fontsize=6)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1.5, labelsize=5.6)
    fig.text(0.055, 0.955, "Cross-compartment coordination loosens at PPD7",
             fontsize=7.5, ha="left", va="top")
    ep_im = mats["PPD7"].loc["Epithelial", "Immune"]
    fig.text(0.055, 0.055, f"PPD7 epithelial-immune falls to {ep_im:.2f}, the lowest "
             "pair in any stage. The diagonal is blank; self-correlation is 1 by "
             "construction.", fontsize=5.6, color=GREY, ha="left", va="bottom")
    return fig


def panel_d(d):
    """Layer retention under three anchors, plus the anchor-agreement summary."""
    an, ag = d["anchor"], d["agree"].set_index("layer")
    fig = plt.figure(figsize=(188 * MM, 152 * MM))
    x = np.arange(len(STAGES5))

    handles = None
    for k, layer in enumerate(LAYERS):
        row, col = divmod(k, 3)
        ax = fig.add_axes([0.085 + col * 0.305, 0.665 - row * 0.275, 0.235, 0.185])
        for anc in ANCHORS:
            sub = an[an.anchor == anc].set_index("stage")
            ax.plot(x, [sub[layer][s] for s in STAGES5], color=STAGE_COL[anc], lw=1.4,
                    marker="o", ms=3.0, label=STAGE_LAB[anc])
            ax.plot([STAGES5.index(anc)], [0.0], marker="o", ms=5.2, mfc="none",
                    mec=STAGE_COL[anc], mew=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([STAGE_LAB[s] for s in STAGES5], rotation=45, ha="right")
        ax.set_ylim(-0.08, 1.12)
        ax.set_yticks([0, 0.5, 1.0])
        ax.set_title(layer, fontsize=7, pad=3)
        if col == 0:
            ax.set_ylabel("retention (scaled\nwithin anchor)")
        if k == 0:
            handles = ax.get_legend_handles_labels()

    # The anchor key sits on the figure, not inside the first sub-axes: every one of the
    # six has data in all four corners.
    seen = {lab: h for h, lab in zip(*handles)}
    fig.legend([seen[STAGE_LAB[a]] for a in ANCHORS], [STAGE_LAB[a] for a in ANCHORS],
               title="anchor", loc="upper right",
               bbox_to_anchor=(0.995, 0.995), ncol=3, fontsize=6, title_fontsize=6,
               handlelength=1.4, columnspacing=1.0, borderaxespad=0.0)

    axs = fig.add_axes([0.085, 0.115, 0.735, 0.155])
    xb = np.arange(len(LAYERS))
    width = 0.26
    for j, (col, lab) in enumerate(AGREE):
        axs.bar(xb + (j - 1) * width, ag[col].loc[LAYERS].values, width * 0.92,
                color=AGREE_COL[j], label=lab, zorder=2)
    for t in (INVARIANCE_THRESHOLD, -INVARIANCE_THRESHOLD):
        axs.axhline(t, color="#b03030", lw=0.7, ls=(0, (3, 2)), zorder=1)
    axs.axhline(0, color="#333333", lw=0.6, zorder=1)
    axs.text(len(LAYERS) - 0.38, INVARIANCE_THRESHOLD, f"$\\pm${INVARIANCE_THRESHOLD:.2f}",
             fontsize=5.6, color="#b03030", va="center")
    comp = ag[AGREE[0][0]]["Composition"]
    axs.annotate(f"{comp:+.2f}", xy=(LAYERS.index("Composition") - width, comp),
                 xytext=(LAYERS.index("Composition") - width, 0.34), fontsize=6,
                 color="#1a1a1a", ha="center", va="bottom",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=GREY, shrinkA=1,
                                 shrinkB=1))
    axs.set_xticks(xb)
    axs.set_xticklabels(LAYERS)
    axs.set_ylim(-1.15, 1.05)
    axs.set_ylabel("agreement between\nanchor choices ($r$)")
    axs.legend(loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, fontsize=5.6,
               handlelength=1.2, columnspacing=1.2, borderaxespad=0.0)
    axs.set_xlim(-0.6, len(LAYERS) - 0.4)

    fig.text(0.085, 0.985, "Layer retention is not anchor invariant", fontsize=7.5,
             ha="left", va="top")
    fig.text(0.085, 0.963, "The same six layers recomputed with CTRL, E16.5 and PPD1 as\n"
             "the anchor. The open ring marks the anchor stage, at 0 by construction.",
             fontsize=5.6, color=GREY, ha="left", va="top", linespacing=1.5)
    fig.text(0.085, 0.055, "Five layers reverse sign between the CTRL and E16.5 anchors, "
             "so no layer reaches +0.75 agreement.\nComposition is the only positive "
             "value and it is near zero: between those two anchors its ordering is\n"
             "unrelated rather than reversed.", fontsize=5.6, color=GREY, ha="left",
             va="top", linespacing=1.5)
    return fig


# The values the pre-final deck prints for this figure. verify_numbers checks the staged
# tables against them so a table swap cannot pass.
EXPECT_BETWEEN = {("E16_5", "PPD1"): 0.675988, ("E16_5", "PPD7"): 0.388457,
                  ("E16_5", "PPD30"): 0.276734, ("PPD1", "PPD7"): 0.574250,
                  ("PPD1", "PPD30"): 0.625065, ("PPD7", "PPD30"): 0.459274}
EXPECT_PPD7_EPI_IMM = 0.160702
EXPECT_SENSITIVITY = {"Anatomy": -1.00, "Identity": -0.77, "Composition": 0.09,
                      "State": -0.84, "Compartment": -0.40, "Signaling": -0.98}


def verify_numbers(d, tol=1e-6):
    """Check every deck number against the staged tables. Returns a list of rows."""
    rows = []

    def chk(name, got, want, t=tol):
        rows.append((name, want, got, "PASS" if abs(got - want) <= t else "FAIL"))

    bt = d["between"].set_index("stage")
    for (a, b), want in EXPECT_BETWEEN.items():
        chk(f"between-stage {a} vs {b}", float(bt.loc[a, b]), want)
        chk(f"between-stage symmetry {b} vs {a}", float(bt.loc[b, a]), float(bt.loc[a, b]))
    for s in STAGES:
        chk(f"between-stage diagonal {s}", float(bt.loc[s, s]), 1.0)

    pw = d["pairwise"]
    m7 = _matrix(pw, "PPD7")
    chk("PPD7 epithelial-immune", float(m7.loc["Epithelial", "Immune"]),
        EXPECT_PPD7_EPI_IMM)
    lowest = pw.loc[pw.pearson_r.idxmin()]
    rows.append(("lowest pair across all stages", "PPD7 Epithelial-Immune",
                 f"{lowest.stage} {lowest.compartment_a}-{lowest.compartment_b}",
                 "PASS" if (lowest.stage == "PPD7" and {lowest.compartment_a,
                            lowest.compartment_b} == {"Epithelial", "Immune"}) else "FAIL"))
    pc = d["percomp"]
    for s in STAGES:
        for c in COMPS:
            got = float(pc[(pc.stage == s) & (pc.compartment == c)].mean_r_to_others.iloc[0])
            sub = pw[(pw.stage == s) & ((pw.compartment_a == c) | (pw.compartment_b == c))]
            chk(f"mean_r_to_others = mean of 3 pairs, {s} {c}", got,
                float(sub.pearson_r.mean()), 1e-9)
    ws = d["within"].set_index("stage")
    for s in STAGES:
        chk(f"within_stage_r = mean of 6 pairs, {s}", float(ws.within_stage_r[s]),
            float(pw[pw.stage == s].pearson_r.mean()), 1e-9)

    ag = d["agree"].set_index("layer")
    for layer, want in EXPECT_SENSITIVITY.items():
        chk(f"anchor sensitivity {layer}", round(float(ag[AGREE[0][0]][layer]), 2), want,
            5e-3)
    for layer in LAYERS:
        chk(f"anchor CTRL_range {layer}", float(ag.CTRL_range[layer]), 1.0)

    an = d["anchor"]
    for anc in ANCHORS:
        sub = an[an.anchor == anc].set_index("stage")
        for layer in LAYERS:
            chk(f"anchor {anc} self-retention {layer}", float(sub[layer][anc]), 0.0, 1e-9)
            chk(f"anchor {anc} max retention {layer}", float(sub[layer].max()), 1.0, 1e-9)
    for layer in LAYERS:
        for col, lab in [("CTRL_PPD30", "CTRL"), ("PPD1_PPD30", "PPD1"),
                         ("E16_5_PPD30", "E16_5")]:
            got = float(ag[col][layer])
            want = float(an[(an.anchor == lab) & (an.stage == "PPD30")][layer].iloc[0])
            chk(f"agreement {col} matches by-anchor PPD30, {layer}", got, want, 1e-9)
    return rows


def invariance_call(d, threshold=INVARIANCE_THRESHOLD):
    """Which layers clear the anchor-invariance threshold, signed and absolute.

    Reported both ways because the sign carries the finding: a signed test asks whether
    the two anchors agree, an absolute test asks only whether they are strongly related,
    and a layer at r = -1.00 passes the second while failing the first.
    """
    ag = d["agree"].set_index("layer")[AGREE[0][0]]
    return {layer: {"r": float(ag[layer]),
                    "clears_signed": bool(ag[layer] >= threshold),
                    "clears_absolute": bool(abs(ag[layer]) >= threshold),
                    "sign": "positive" if ag[layer] > 0 else "negative"}
            for layer in LAYERS}


_PANELS = {
    "a": (panel_a, "extra_coordination_per_compartment_vs_null"),
    "b": (panel_b, "extra_between_stage_similarity"),
    "c": (panel_c, "extra_cross_compartment_matrices"),
    "d": (panel_d, "extra_layer_retention_by_anchor"),
}


def build(repo_root, out_dir):
    """Render SFig5 panels a-d; returns {panel: svg_path}."""
    d = load_inputs(repo_root)
    bad = [r for r in verify_numbers(d) if r[3] == "FAIL"]
    if bad:
        raise ValueError("SFig5 source tables disagree with the deck: "
                         + "; ".join(f"{n}: expected {w}, got {g}" for n, w, g, _ in bad))
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    out = {}
    for label, (fn, stem) in _PANELS.items():
        fig = fn(d)
        path = os.path.join(out_dir, stem)
        render_svg.save_svg(fig, path, also_pdf=True)
        v = render_svg.verify(path + ".svg")
        if not v["text_elements"]:
            raise RuntimeError(f"SFig5{label} exported with no live text: {v}")
        plt.close(fig)
        out[label] = path + ".svg"
    return out
