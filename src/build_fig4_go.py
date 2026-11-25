"""Fig 4 GO dotplot, five wave bands, recomputed from the wave gene sets.

Downstream of the heatmap: the wave sets define the rows, so this must be re-run whenever
the module/wave table changes. Selection rule (kept from the original panel): per wave, the
N terms unique to that wave with the smallest adjusted P, ordered by ascending gene ratio.
"""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt

ABBR = [
    ("Mitotic Sister Chromatid Segregation", "Mitotic sister chromatid segregation"),
    ("DNA-templated DNA Replication Maintenance Of Fidelity", "DNA replication fidelity"),
    ("Positive Regulation Of Response To External Stimulus", "Pos. reg. response to external stimulus"),
    ("Positive Regulation Of Epithelial To Mesenchymal Transition", "Pos. reg. epithelial\u2013mesenchymal transition"),
    ("Positive Regulation Of Cell Differentiation", "Pos. reg. cell differentiation"),
    ("Cardiac Muscle Cell-Cardiac Muscle Cell Adhesion", "Muscle-cell adhesion"),
    ("Regulation Of Ventricular Cardiac Muscle Cell Action Potential",
     "Reg. muscle action potential"),
    ("Regulation Of Heart Rate By Cardiac Conduction", "Reg. rate by muscle conduction"),
    ("Negative Regulation Of Cytokine-Mediated Signaling Pathway",
     "Neg. reg. cytokine-mediated signalling"),
    ("Antigen Receptor-Mediated Signaling Pathway", "Antigen receptor-mediated signalling"),
    ("T Cell Receptor Signaling Pathway", "TCR signalling pathway"),
    ("B Cell Receptor Signaling Pathway", "BCR signalling pathway"),
    ("Regulation Of Lymphocyte Activation", "Reg. lymphocyte activation"),
    ("Regulation Of Inflammatory Response", "Reg. inflammatory response"),
    ("Regulation Of Blood Circulation", "Reg. blood circulation"),
    ("Regulation Of Heart Contraction", "Reg. heart contraction"),
    ("Intermediate Filament Organization", "Intermediate filament organization"),
    ("DNA Replication Initiation", "DNA replication initiation"),
    ("DNA Metabolic Process", "DNA metabolic process"),
    ("Mitotic Nuclear Division", "Mitotic nuclear division"),
    ("Sister Chromatid Segregation", "Sister chromatid segregation"),
    ("Collagen Fibril Organization", "Collagen fibril organization"),
    ("Inflammatory Response", "Inflammatory response"),
    ("Neutrophil Chemotaxis", "Neutrophil chemotaxis"),
    ("Neutrophil Migration", "Neutrophil migration"),
    ("Granulocyte Chemotaxis", "Granulocyte chemotaxis"),
    ("Gamma-Delta T Cell Activation", "\u03b3\u03b4 T cell activation"),
    ("T Cell Activation", "T cell activation"),
    ("Cardiac Conduction", "Muscle conduction"),
    ("Eye Development", "Eye development"),
]


def tidy(term):
    """Strip the GO id and apply the abbreviation table (longest match first)."""
    base = term.split(" (GO:")[0].strip()
    for long_, short_ in sorted(ABBR, key=lambda p: -len(p[0])):
        if base.lower() == long_.lower():
            return short_
    return base


def select(sig, waves, n_per=6):
    """Per wave: the n_per wave-unique terms with smallest adj-P, drawn bottom-up by ratio."""
    out = []
    for w in waves:
        s = sig[(sig.wave == w) & sig["unique"]].nsmallest(n_per, "Adjusted P-value")
        s = s.sort_values("gene_ratio", ascending=False)
        out.append(s)
    return pd.concat(out, ignore_index=True)


def build(sel, waves, wcol, wpeak, stage_lab, outstem, title, group_word="module",
          fs=1.0, save=None, row_scale=1.0, dot_scale=None, x_scale=1.0):
    n = len(sel)
    ROW_H = 0.208*row_scale
    TOP, BOT = 0.52*fs, 1.62*fs
    LEFT, PLOTW, RIGHT = 3.05*fs, 4.05*max(1.0, fs*0.72)*x_scale, 1.05*fs
    FIG_W, FIG_H = LEFT + PLOTW + RIGHT, TOP + n*ROW_H + BOT
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fx, fy = (lambda v: v/FIG_W), (lambda v: v/FIG_H)
    ax = fig.add_axes([fx(LEFT), fy(BOT), fx(PLOTW), fy(n*ROW_H)])

    y = np.arange(n)[::-1]
    lo, hi = sel.gene_ratio.min(), sel.gene_ratio.max()
    pad = 0.06*(hi-lo)
    # wave bands, tinted with the wave colour
    start = 0
    for w in waves:
        k = int((sel.wave == w).sum())
        if not k:
            continue
        y0, y1 = y[start+k-1]-0.5, y[start]+0.5
        ax.axhspan(y0, y1, color=wcol[w], alpha=0.07, lw=0, zorder=0)
        ax.text(1.012, (y0+y1)/2, f"{w}\n{stage_lab[wpeak[w]]}", transform=ax.get_yaxis_transform(),
                ha="left", va="center", fontsize=7.6*fs, color=wcol[w], fontweight="bold",
                linespacing=1.35)
        start += k

    # Dot area is set by dot_scale, not by row_scale: tightening the rows must not
    # shrink the markers, or 'genes in term' stops being comparable to the size key.
    DS = fs**1.6 if dot_scale is None else dot_scale
    sizes = (4.2*sel.nGenes.values + 18) * DS
    sc = ax.scatter(sel.gene_ratio, y, s=sizes, c=sel.negLog10, cmap="viridis",
                    edgecolors="none", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([tidy(t) for t in sel.Term], fontsize=7.0*fs)
    for lab, w in zip(ax.get_yticklabels(), sel.wave):
        lab.set_color(wcol[w])
    ax.set_xlim(lo-pad, hi+pad); ax.set_ylim(-0.6, n-0.4)
    ax.set_xlabel(f"gene ratio ({group_word} genes in term / term size)", fontsize=7.6*fs)
    ax.tick_params(axis="x", labelsize=7.0*fs, length=2.5*fs)
    ax.tick_params(axis="y", length=0)
    for s_ in ("top", "right"): ax.spines[s_].set_visible(False)
    for s_ in ("left", "bottom"): ax.spines[s_].set_linewidth(0.6)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e8e8e8", lw=0.5*fs, zorder=1)

    # colourbar
    # The legend band is laid out on the FULL figure width, not the plot column: when
    # x_scale narrows the plot, a PLOTW-relative clamp goes negative and add_axes dies.
    BAND_L, BAND_R = 0.55*fs, FIG_W - 0.35*fs
    KEY_W_ = 1.62*fs
    CB_W = max(0.9*fs, (BAND_R - BAND_L) - KEY_W_ - 0.70*fs)
    cax = fig.add_axes([fx(BAND_L), fy(0.62*fs), fx(CB_W), fy(0.085*fs)])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.5); cax.tick_params(labelsize=6.4*fs, length=2*fs)
    cb.set_label("\u2212log10 adjusted P", fontsize=6.8*fs, labelpad=2*fs)
    # size key, one axes wide enough for the largest marker, so it cannot clip to a square
    ref = [5, 20, 45]
    KEY_W, KEY_H = 1.62*fs, 0.34*fs
    xs = BAND_R - KEY_W
    ka = fig.add_axes([fx(xs), fy(0.56*fs), fx(KEY_W), fy(KEY_H)]); ka.set_axis_off()
    ka.set_xlim(0, 1); ka.set_ylim(0, 1)
    xpos = [0.16, 0.5, 0.84]
    ka.scatter(xpos, [0.62]*3, s=[(4.2*r+18)*DS for r in ref], c="#8a8a8a", edgecolors="none")
    for xp, r in zip(xpos, ref):
        ka.text(xp, 0.10, str(r), ha="center", va="center", fontsize=6.4*fs)
    fig.text(fx(xs + KEY_W/2), fy(0.56*fs+KEY_H+0.02*fs), "genes in term", ha="center", va="bottom",
             fontsize=6.8*fs)

    fig.text(fx(0.10), fy(FIG_H-0.12*fs), title, ha="left", va="top", fontsize=8.6*fs)
    fig.text(fx(0.10), fy(FIG_H-0.30*fs),
             f"terms unique to one {group_word}, top 6 per {group_word} by adjusted P",
             ha="left", va="top", fontsize=6.8*fs, color="#777777")
    for ext in ("svg", "pdf", "png"):
        fig.savefig(f"{outstem}.{ext}", dpi=300 if ext == "png" else None,
                    bbox_inches="tight", transparent=(ext != "png"))
    return fig


# Aggressive abbreviations used only when a target box forces a shallow bottom margin.
# Every entry keeps the biological reading; nothing is dropped, only contracted.
SHORT_LONG = {
    "Pos. reg. epithelial\u2013mesenchymal transition": "Pos. reg. EMT",
    "Pos. reg. response to external stimulus": "Pos. reg. resp. to ext. stimulus",
    "Neg. reg. cytokine-mediated signalling": "Neg. reg. cytokine signalling",
    "Mitotic sister chromatid segregation": "Mitotic sis. chromatid segreg.",
    "Antigen receptor-mediated signalling": "Antigen receptor signalling",
    "Intermediate filament organization": "Interm. filament organization",
    "Reg. rate by muscle conduction": "Reg. rate, muscle conduction",
    "Sister chromatid segregation": "Sister chromatid segreg.",
    "Reg. muscle action potential": "Reg. muscle action pot.",
    "Collagen fibril organization": "Collagen fibril organiz.",
    "Pos. reg. cell differentiation": "Pos. reg. cell diff.",
    "DNA replication initiation": "DNA replication init.",
    "Reg. inflammatory response": "Reg. inflamm. response",
    "Reg. lymphocyte activation": "Reg. lymphocyte activ.",
}


def tidy_short(term, max_chars=None):
    """tidy() plus optional contraction; raises if a label still exceeds max_chars."""
    s = tidy(term)
    if max_chars is None:
        return s
    s = SHORT_LONG.get(s, s)
    if len(s) > max_chars:
        raise ValueError(f"label still {len(s)} chars > {max_chars}: {s!r} "
                         f"- add an entry to SHORT_LONG rather than truncating")
    return s


def build_horizontal(sel, waves, wcol, wpeak, stage_lab, outstem, title, group_word="module",
                     fs=1.0, save=None, col_scale=1.0, dot_scale=None, rotation=45,
                     max_label_chars=None, legend_side=False, fit_box=None,
                     legend_scale=1.0):
    """Landscape transpose of the dotplot: terms run along x, gene ratio up y.

    SAVE path warning: pass a `save` callback that does not use bbox_inches='tight'.
    A tight bbox crops the canvas to the drawn ink, so a figure sized with fit_box saves
    SMALLER than the box it was solved for (observed: 991.59x430 pt solved, 978.3x405.5 pt
    saved). Always confirm the SAVED svg's own width/height, never fig.get_size_inches().

    Same encoding as build(), dot area = genes in term, colour = -log10 adj P, module
    bands shaded, but the long axis is now horizontal, so the term names move into the
    bottom margin and are angled. That margin is the cost of the layout: the longest term
    is ~43 characters, which needs several inches of depth at 2x type. Everything else
    (dot areas, the size key, colour normalisation) is shared with the vertical version so
    the two are directly comparable.
    """
    import numpy as np, matplotlib.pyplot as plt

    n = len(sel)
    COL_W = 0.62 * col_scale * max(1.0, fs * 0.55)
    # y-label is two rotated lines + short tick labels; 1.30*fs over-reserved badly at 2x
    LEFT, RIGHT = 0.52 * fs + 0.42, 0.16 * fs
    TOP, HDR = 0.52 * fs, 0.46 * fs
    PLOTH = 3.30 * max(1.0, fs * 0.62)
    # Bottom margin is set by the LONGEST term name at this type size and angle; measured
    # empirically at 0.047 in per character per fs unit, times sin(rotation).
    labels = [tidy_short(t, max_label_chars) for t in sel.Term]
    LAB_H = 0.047 * fs * max(len(s) for s in labels) * (
        1.0 if rotation >= 90 else np.sin(np.radians(rotation))) + 0.10 * fs
    KEY_H = 0.0 if legend_side else 0.55 * fs
    LS = legend_scale
    LEG_W = ((1.62 * fs + 0.52 * fs + 0.62) * LS) if legend_side else 0.0
    PLOTW = n * COL_W
    FIG_W = LEFT + PLOTW + RIGHT + LEG_W
    if fit_box is not None:      # honour an exact target box: solve PLOTW/PLOTH for it
        FIG_W = fit_box[0]
        PLOTW = FIG_W - LEFT - RIGHT - LEG_W
        COL_W = PLOTW / n
    FIG_H = TOP + HDR + PLOTH + LAB_H + KEY_H
    if fit_box is not None:
        FIG_H = fit_box[1]
        PLOTH = FIG_H - TOP - HDR - LAB_H - KEY_H
        if PLOTH <= 0.6:
            raise ValueError(f'fit_box height {fit_box[1]:.2f} in leaves PLOTH={PLOTH:.2f} in; '
                             f'shorten labels (max_label_chars) or lower fs')
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fx, fy = (lambda v: v / FIG_W), (lambda v: v / FIG_H)
    y0 = KEY_H + LAB_H
    ax = fig.add_axes([fx(LEFT), fy(y0), fx(PLOTW), fy(PLOTH)])

    DS = fs ** 1.6 if dot_scale is None else dot_scale
    sizes = (4.2 * sel.nGenes.values + 18) * DS
    x = np.arange(n)

    # module bands as vertical spans, in course order
    start = 0
    for w in waves:
        k = int((sel.wave == w).sum())
        if k == 0:
            continue
        ax.axvspan(start - 0.5, start + k - 0.5, color=wcol[w], alpha=0.075, lw=0, zorder=0)
        fig.text(fx(LEFT + (start + k / 2) * COL_W), fy(y0 + PLOTH + 0.10 * fs),
                 f"{w}\n{stage_lab.get(wpeak[w], wpeak[w])}", ha="center", va="bottom",
                 fontsize=7.6 * fs, color=wcol[w], fontweight="bold", linespacing=1.25)
        start += k

    sc = ax.scatter(x, sel.gene_ratio.values, s=sizes, c=sel.negLog10.values,
                    cmap="viridis", edgecolors="none", zorder=3)
    ax.grid(axis="y", color="#e8e8e8", lw=0.5 * fs, zorder=1)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.7, n - 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=rotation,
                       ha="right" if 0 < rotation < 90 else "center",
                       rotation_mode="anchor" if 0 < rotation < 90 else None,
                       fontsize=7.0 * fs)
    for t, w in zip(ax.get_xticklabels(), sel.wave):
        t.set_color(wcol[w])
    ax.set_ylabel(f"gene ratio\n({group_word} genes in term / term size)", fontsize=7.6 * fs)
    ax.tick_params(axis="y", labelsize=7.0 * fs, length=2.5 * fs)
    ax.tick_params(axis="x", length=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)

    # legend band along the bottom, full figure width
    ref = [5, 20, 45]
    if legend_side:
        # Width is the plentiful axis in a wide target box and height is not, so the two
        # keys stack in the right gutter instead of eating a footer strip.
        gx = LEFT + PLOTW + RIGHT + 0.10
        KEY_W = 1.62 * fs * LS
        # colourbar tick labels sit to the RIGHT of the bar, so the size-key title cannot
        # share that x. Offset the size key clear of the colourbar label column.
        kx = gx + 0.52 * fs * LS
        cax = fig.add_axes([fx(gx), fy(y0 + PLOTH * 0.56), fx(0.10 * fs * LS), fy(PLOTH * 0.42 * LS)])
        cb = fig.colorbar(sc, cax=cax)
        cb.outline.set_linewidth(0.5)
        cax.tick_params(labelsize=6.4 * fs * LS, length=2 * fs * LS)
        cb.set_label("\u2212log10 adjusted P", fontsize=6.8 * fs * LS, labelpad=3 * fs * LS)
        # The size key needs enough HEIGHT for three stacked markers plus their gaps:
        # the largest is `dmax` points across, so the axes must be at least 3.4x that or
        # the markers and their labels collide into a single blob.
        # legend_scale shrinks text and bar thickness only. The size-key markers must stay
        # at the data dot area (KDS = DS) or the key stops measuring the dots it explains.
        KDS = DS
        dmax = 2 * (((4.2 * max(ref) + 18) * KDS) / np.pi) ** 0.5
        key_h = max(PLOTH * 0.30 * LS, 3.4 * dmax / 72)
        ka = fig.add_axes([fx(kx), fy(y0), fx(KEY_W), fy(key_h)])
        ka.set_axis_off(); ka.set_xlim(0, 1); ka.set_ylim(0, 1)
        ypos = [0.78, 0.45, 0.13]
        ka.scatter([0.16] * 3, ypos, s=[(4.2 * r + 18) * KDS for r in ref], c="#8a8a8a",
                   edgecolors="none")
        for yp, r in zip(ypos, ref):
            ka.text(0.40, yp, str(r), ha="left", va="center", fontsize=6.4 * fs * LS)
        fig.text(fx(kx), fy(y0 + key_h + 0.04), "genes in term", ha="left", va="bottom",
                 fontsize=6.8 * fs * LS)
        fig.text(fx(0.10), fy(FIG_H - 0.12 * fs), title, ha="left", va="top",
                 fontsize=8.6 * fs)
        fig.text(fx(0.10), fy(FIG_H - 0.30 * fs),
                 f"terms unique to one {group_word}, top 6 per {group_word} by adjusted P",
                 ha="left", va="top", fontsize=6.8 * fs, color="#777777")
        if save is not None:
            save(fig, outstem)
        return fig

    BAND_L, BAND_R = LEFT, FIG_W - RIGHT
    KEY_W = 1.62 * fs
    CB_W = max(0.9 * fs, (BAND_R - BAND_L) - KEY_W - 0.90 * fs)
    cax = fig.add_axes([fx(BAND_L), fy(0.30 * fs), fx(CB_W), fy(0.085 * fs)])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    cb.outline.set_linewidth(0.5)
    cax.tick_params(labelsize=6.4 * fs, length=2 * fs)
    cb.set_label("\u2212log10 adjusted P", fontsize=6.8 * fs, labelpad=2 * fs)

    ka = fig.add_axes([fx(BAND_R - KEY_W), fy(0.24 * fs), fx(KEY_W), fy(0.34 * fs)])
    ka.set_axis_off(); ka.set_xlim(0, 1); ka.set_ylim(0, 1)
    xpos = [0.16, 0.5, 0.84]
    ka.scatter(xpos, [0.62] * 3, s=[(4.2 * r + 18) * DS for r in ref], c="#8a8a8a",
               edgecolors="none")
    for xp, r in zip(xpos, ref):
        ka.text(xp, 0.10, str(r), ha="center", va="center", fontsize=6.4 * fs)
    fig.text(fx(BAND_R - KEY_W / 2), fy(0.24 * fs + 0.36 * fs), "genes in term",
             ha="center", va="bottom", fontsize=6.8 * fs)

    fig.text(fx(0.10), fy(FIG_H - 0.12 * fs), title, ha="left", va="top", fontsize=8.6 * fs)
    fig.text(fx(0.10), fy(FIG_H - 0.30 * fs),
             f"terms unique to one {group_word}, top 6 per {group_word} by adjusted P",
             ha="left", va="top", fontsize=6.8 * fs, color="#777777")
    if save is not None:
        save(fig, outstem)
    return fig
