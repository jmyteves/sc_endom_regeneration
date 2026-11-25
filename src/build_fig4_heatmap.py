"""Fig 4, gene-level phase-signature heatmap, rows grouped by PHASE (not by raw module id).

Rebuild of the Fig3 phase-signature heatmap with the row side-annotation collapsed from
six k-means modules to the five named phases of the postpartum course.

Drop-in for the Fig 3 remake: point MODULE_CSV at the new module table in
sc_endom_EDITABLE_SVG/fig3_remake/modules_in/. The table needs columns
    gene, module (required)
    phase (optional, if absent, PHASE_MAP below is used)
Everything else (column order, tracks, z-scoring) is unchanged from the original panel,
so the new panel is visually comparable to the one it replaces.
"""
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
import matplotlib.colors as mcolors

STAGES = ["CTRL", "E16_5", "PPD1", "PPD7", "PPD30"]
LABD   = {"CTRL": "CTRL", "E16_5": "E16.5", "PPD1": "PPD1", "PPD7": "PPD7", "PPD30": "PPD30"}
# Fig3 v2 five-wave palette (user-directed 17 Aug). One colour per stage, and therefore
# one colour per phase, since each phase owns exactly one peak stage. The stage labels on
# the x-axis and the phase bars on the side are drawn from this single dict, which is what
# makes the side annotation consistent with the timeline.
PAL_D  = dict(zip(STAGES, ['#808080', '#cb7a29', '#A0522D', '#FA8072', '#4682B4']))
C4O    = ["Epithelial", "Stromal", "Endothelial", "Immune"]
P4D    = {"Epithelial": "#708090", "Stromal": "#008080", "Endothelial": "#fa8072", "Immune": "#cd853f"}
C4 = {"Epithelial": ["Epithelial"],
      "Stromal": ["Fibroblast", "Fibroblast (Igf1+)", "Stromal (remodeling)", "Proliferating stromal",
                  "Pericyte", "Smooth muscle", "Mesothelial"],
      "Endothelial": ["Vascular endothelial", "Lymphatic endothelial"],
      "Immune": ["Macrophage", "Macrophage/APC", "Dendritic cell", "Dendritic cell (cDC1)",
                 "T cell", "NK cell"]}
t2c4 = {t: c for c, ts in C4.items() for t in ts}
DECK16 = {'Fibroblast': '#008080', 'Stromal (remodeling)': '#1e90ff', 'Proliferating stromal': '#483d8b',
          'Fibroblast (Igf1+)': '#4682b4', 'Smooth muscle': '#556b2f', 'Pericyte': '#20b2aa',
          'Vascular endothelial': '#fa8072', 'Lymphatic endothelial': '#8b4513',
          'Mesothelial': '#db7093', 'Epithelial': '#708090', 'Macrophage': '#cd853f',
          'Dendritic cell': '#ff8c00', 'Macrophage/APC': '#ffd700',
          'Dendritic cell (cDC1)': '#8b0000', 'T cell': '#9370db', 'NK cell': '#c71585'}
types16 = sorted(t2c4.keys())
short = {t: (t.replace("Dendritic cell (cDC1)", "cDC1").replace("Dendritic cell", "DC")
             .replace("Vascular endothelial", "Vasc. endo.").replace("Lymphatic endothelial", "Lymph. endo.")
             .replace("Stromal (remodeling)", "Stromal (remod.)").replace("Fibroblast (Igf1+)", "Fib. Igf1+")
             .replace("Proliferating stromal", "Prolif. stromal").replace("Macrophage/APC", "Mac/APC")
             .replace("Smooth muscle", "Sm. muscle")) for t in types16}

# ---- the five phases, in course order. modules = which k-means module ids belong to each.
PHASE_ORDER = ["Homeostasis", "Pregnancy", "Early repair", "Immune resolution", "Barrier reformation"]
PHASE_MAP = {"Homeostasis": ["M4", "M5"], "Pregnancy": ["M0"], "Early repair": ["M2"],
             "Immune resolution": ["M1"], "Barrier reformation": ["M3"]}
PHASE_PEAK = {"Homeostasis": "CTRL", "Pregnancy": "E16_5", "Early repair": "PPD1",
              "Immune resolution": "PPD7", "Barrier reformation": "PPD30"}
# Alternative names for the same five stages (Fig3 v2 wave vocabulary, earlier whiteboard
# wording). Mapping a phase name to its peak stage is what drives BOTH the row order and
# the colour, so every consumer below goes through phase_stage() rather than reading
# PHASE_PEAK directly, otherwise an aliased name colours correctly but sorts to the end.
PHASE_ALIAS = {"Late repair": "PPD7", "Late repair / early homeostasis": "PPD7",
               "Immune resolution": "PPD7",
               "Remodelling": "PPD30", "Remodeling": "PPD30", "Barrier reformation": "PPD30",
               "Late homeostasis / remodelling": "PPD30"}


def phase_stage(phase):
    """The stage a phase peaks in, canonical name or any accepted alias. None if unknown."""
    return PHASE_PEAK.get(phase) or PHASE_ALIAS.get(phase)


def phase_colour(phase):
    """A phase is coloured by the stage it peaks in, so side bars match the x-axis labels."""
    return PAL_D.get(phase_stage(phase), "#999999")


def load_inputs(npz_path, module_csv, colann_csv, state_csv):
    z = np.load(npz_path, allow_pickle=True)
    levs = [str(x) for x in z["levels"]]
    genes = [str(g) for g in z["genes"]]
    M = z["sums"].astype(float)
    tot = M.sum(1, keepdims=True); tot[tot == 0] = 1
    Erep = pd.DataFrame(M / tot * 1e6, index=levs, columns=genes)
    mods = pd.read_csv(module_csv)
    cann = pd.read_csv(colann_csv)
    stc = pd.read_csv(state_csv)
    return Erep, mods, cann, stc


def phase_blocks(mods):
    """Order genes by phase (course order), then by module within phase. Returns (gene_order, blocks)."""
    if "phase" in mods.columns:
        ph_of = dict(zip(mods.module.unique(),
                         mods.groupby("module")["phase"].first().reindex(mods.module.unique())))
    else:
        ph_of = {m: p for p, ms in PHASE_MAP.items() for m in ms}
    unknown = sorted({m for m in mods.module.unique() if m not in ph_of})
    if unknown:
        raise ValueError(f"modules with no phase assignment: {unknown}: extend PHASE_MAP "
                         f"or add a 'phase' column to the module table")
    # Order by the stage each phase peaks in, resolving aliases, so a table written in any
    # accepted vocabulary sorts into course order. Names with no known stage go last, in
    # encounter order, and are reported so they can't be mistaken for correctly placed rows.
    present = list(dict.fromkeys(ph_of.values()))
    staged = [p for p in present if phase_stage(p) is not None]
    unstaged = [p for p in present if phase_stage(p) is None]
    order = sorted(staged, key=lambda p: STAGES.index(phase_stage(p)))
    order += unstaged
    if unstaged:
        print(f"[phase_blocks] no known peak stage for {unstaged}: appended after the "
              f"ordered phases and coloured grey; add them to PHASE_ALIAS to place them.")
    gene_order, blocks, p0 = [], [], 0
    for ph in order:
        ms = [m for m in mods.module.unique() if ph_of[m] == ph]
        sub, sub_ms = [], []
        for m in ms:
            g = list(mods.loc[mods.module == m, "gene"])
            sub += g; sub_ms.append((m, len(g)))
        gene_order += sub
        blocks.append(dict(phase=ph, start=p0, end=p0+len(sub), n=len(sub), modules=sub_ms))
        p0 += len(sub)
    return gene_order, blocks


def build(Erep, mods, cann, stc, outstem, title, subtitle=None, fs=1.0, save=None):
    gene_order, blocks = phase_blocks(mods)
    rep2stage = {k.split("|")[0]: "_".join(k.split("|")[0].split("_")[:-1]) for k in Erep.index}
    col_order = sorted(list(cann.col), key=lambda c: (STAGES.index(rep2stage[c.split("|")[0]]),
                                                      C4O.index(t2c4[c.split("|", 1)[1]]), c))
    cs = [rep2stage[c.split("|")[0]] for c in col_order]
    cc = [t2c4[c.split("|", 1)[1]] for c in col_order]
    ctypes = [c.split("|", 1)[1] for c in col_order]
    runs, prev = [], None
    for i, s_ in enumerate(cs):
        if s_ != prev: runs.append([s_, i, i+1]); prev = s_
        else: runs[-1][2] = i+1
    bounds_x = [(a, b, s_) for s_, a, b in runs]
    E = np.log1p(Erep.loc[col_order, gene_order])
    H = ((E - E.mean()) / E.std().replace(0, np.nan)).T
    NG = len(gene_order)

    plt.close("all")
    mpl.rcParams.update({"font.size": 9.5*fs, "axes.labelsize": 9.5*fs, "axes.titlesize": 10.5*fs,
                         "xtick.labelsize": 8.5*fs, "ytick.labelsize": 8.5*fs, "legend.fontsize": 8.5*fs})
    # canvas grows sub-linearly with type: the matrix itself must not balloon, but the
    # row-label column and the legend block need real room at 2x.
    W_SC = 1.0 + 0.26*(fs-1.0)
    H_SC = 1.0 + 0.34*(fs-1.0)
    fig = plt.figure(figsize=(12.6*W_SC, 11.3*H_SC))
    HEAT_H_IN = 11.3*H_SC * (1/(1+0.030*2)) * 0.66   # approx height of the matrix axes, for label fitting
    # two column tracks only: the state track is dropped, the row blocks ARE the state.
    gsm = fig.add_gridspec(3, 2, height_ratios=[0.030, 0.030, 1],
                           width_ratios=[0.075*(1+0.85*(fs-1.0)), 1], hspace=0.010, wspace=0.014,
                           top=0.905, bottom=0.215+0.055*(fs-1.0), left=0.075+0.030*(fs-1.0), right=0.945)
    for i, (lab_, vals, cm_, vmx) in enumerate([
            ("clusters", [types16.index(t) for t in ctypes],
             mcolors.ListedColormap([DECK16[t] for t in types16]), len(types16)-1),
            ("compartment", [C4O.index(c) for c in cc],
             mcolors.ListedColormap([P4D[c] for c in C4O]), 3)]):
        A = fig.add_subplot(gsm[i, 1])
        A.imshow([vals], aspect="auto", cmap=cm_, vmin=0, vmax=vmx)
        A.set_xticks([]); A.set_yticks([])
        A.text(-1.5, 0, lab_, fontsize=8.6*fs, ha="right", va="center", color="#333")
        for a, b, s_ in bounds_x[:-1]: A.axvline(b-0.5, color="white", lw=1.4)

    axM = fig.add_subplot(gsm[2, 1]); axR = fig.add_subplot(gsm[2, 0])
    axM.imshow(H.values, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2, interpolation="nearest")
    axM.set_xticks([]); axM.set_yticks([])
    for blk in blocks[:-1]: axM.axhline(blk["end"]-0.5, color="white", lw=2.2)
    # thin sub-rules where two modules share a phase
    for blk in blocks:
        off = blk["start"]
        for m, n in blk["modules"][:-1]:
            off += n
            axM.axhline(off-0.5, color="white", lw=0.7, alpha=0.85)
    for a, b, s_ in bounds_x[:-1]: axM.axvline(b-0.5, color="white", lw=1.6)
    for a, b, s_ in bounds_x:
        axM.text((a+b)/2, NG*1.012, LABD[s_], fontsize=10.0*fs, ha="center", va="top",
                 color=PAL_D[s_], fontweight="bold", clip_on=False)
    axM.annotate("", xy=(len(col_order)*0.60, NG*1.055), xytext=(len(col_order)*0.40, NG*1.055),
                 annotation_clip=False, arrowprops=dict(arrowstyle="-|>", color="#555", lw=1.4))
    axM.text(len(col_order)*0.5, NG*1.083, "postpartum stage", fontsize=9.2*fs, ha="center",
             va="center", color="#555", clip_on=False)
    axM.text(len(col_order)/2, NG*1.110,
             f"each column = one replicate \u00d7 cell type   (n = {len(col_order)}, \u226550 cells)",
             fontsize=7.8*fs, ha="center", va="top", color="#999", clip_on=False)

    # ---- row side annotation: one bar per PHASE
    axR.set_xlim(0, 1); axR.set_ylim(NG, 0); axR.axis("off")
    for blk in blocks:
        ph, a, b = blk["phase"], blk["start"], blk["end"]
        col = phase_colour(ph)
        axR.add_patch(plt.Rectangle((0.80, a), 0.20, b-a, color=col, lw=0))
        mid = (a+b)/2
        mtxt = "+".join(m for m, _ in blk["modules"])
        # rotated names must fit the block's own height: wrap onto as many lines as needed
        avail_pt = (b-a)/NG * HEAT_H_IN * 72 * 0.92
        words, lines, fsz = ph.split(), None, 8.0*fs
        for nl in (1, 2, 3):
            per = -(-len(words)//nl)
            cand = [" ".join(words[i:i+per]) for i in range(0, len(words), per)]
            if max(len(s) for s in cand) * fsz * 0.60 <= avail_pt:
                lines = cand; break
        if lines is None:
            lines, fsz = words, 7.0*fs
        axR.text(0.68, mid, "\n".join(lines), fontsize=fsz, ha="center", va="center", color=col,
                 fontweight="bold", rotation=90, linespacing=1.25)
        axR.text(0.22, mid, f"{mtxt}\n{blk['n']} genes", fontsize=6.2*fs, ha="center", va="center",
                 color="#777", rotation=90, linespacing=1.3)
    sm = plt.cm.ScalarMappable(cmap="RdBu_r", norm=mcolors.Normalize(-2, 2))
    cax = axM.inset_axes([1.012, 0.0, 0.014, 0.20])
    cb = fig.colorbar(sm, cax=cax); cb.set_label("z-score", fontsize=7.4*fs)
    cb.ax.tick_params(labelsize=6.6*fs)
    cb.set_ticks([-2, -1, 0, 1, 2])   # 9 ticks crowd the bar at 2x
    fig.suptitle(title, x=0.075, ha="left", fontsize=12*fs, y=0.978)
    if subtitle:
        fig.text(0.075, 0.949, subtitle, ha="left", va="top", fontsize=8.4*fs, color="#666")

    hC = [plt.Line2D([], [], marker="s", ls="", ms=8, color=P4D[c], label=c) for c in C4O]
    hT = [plt.Line2D([], [], marker="s", ls="", ms=7, color=DECK16[t], label=short[t]) for t in types16]
    # Two plain fig.legend() calls. Two rules here, both found in testing:
    # 1. Do not wrap these in fig.add_artist(), that registers a SECOND copy of every
    # legend text, duplicating the strings in the exported SVG.
    # 2. Anchor them at POSITIVE figure coords, inside the canvas. Anchoring below the
    # canvas (negative y) works visually only because savefig(bbox_inches='tight')
    # expands the saved region, but then pre-save text extents do not match the saved
    # output, which makes any geometric overlap check report phantom collisions.
    # Legends inside the canvas keep verification meaningful.
    YL = 0.020 + 0.026/max(1.0, fs*0.85)
    fig.legend(handles=hC, loc="lower left", bbox_to_anchor=(0.02, YL), frameon=False,
               fontsize=8.4*fs, handletextpad=0.4, labelspacing=0.55,
               title="compartment", title_fontsize=8.8*fs)
    fig.legend(handles=hT, loc="lower center", bbox_to_anchor=(0.58, YL), frameon=False,
               fontsize=7.4*fs, ncol=4, handletextpad=0.3, columnspacing=1.4,
               labelspacing=0.5, title="cell identity (16 named types)", title_fontsize=8.8*fs)
    # no phase legend: each phase is named in place on its own row bar, and its colour is
    # the colour of the stage it peaks in, already labelled on the x-axis.

    if save is not None:
        save(fig, outstem)
    else:
        fig.savefig(f"{outstem}.svg", format="svg", transparent=True)
    return fig, blocks, col_order, gene_order
