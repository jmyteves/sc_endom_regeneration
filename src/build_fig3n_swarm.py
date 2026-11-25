"""Fig 3n, per-cell-type swarm matrix, each stage contrasted against control.

Four columns (CTRL vs E16.5 / PPD1 / PPD7 / PPD30) x 16 cell types.
Each row is a beeswarm of per-gene log2FC for that cell type in that contrast.

Colour: the four stage colours own all the hue in this figure (pregnancy amber,
early repair sienna, late repair salmon, remodelling steelblue) for genes UP at
the stage; genes higher in control are grey; non-significant genes pale grey.
Compartment side-bars are therefore a NEUTRAL SLATE RAMP, not the house per-type
palette, because two house type colours (#fa8072, #a0522d) are now stage colours.

Point caps are a DISPLAY subsample only (fixed seed) and never touch the
statistics: the n printed in each gutter is computed from every tested gene.

Called by exec(open(...).read()) from the kernel; expects swarm, sm, ROW_ORDER,
COMP, STG, rsvg, adjust_text in the namespace.
"""
import numpy as np
import matplotlib.pyplot as plt

PAL_STAGE = {"E16.5": "#cb7a29", "PPD1": "#A0522D", "PPD7": "#FA8072", "PPD30": "#4682B4"}
CTRL_GREY = "#8c8c8c"
NS_GREY = "#dcdcdc"
# Compartment swatches use the CANONICAL 4-way compartment palette used by the
# project's compartment UMAP (sampled from that figure): Epithelial #748190,
# Stromal #367f80, Endothelial #e58677, Immune #c58a4c. Two of these sit close to
# stage colours in this figure, Endothelial vs PPD7 dE = 11.8 and Immune vs
# E16.5 dE = 16.4, so the swatch is not the only carrier of compartment: the
# dendrogram groups compartments contiguously and the legend names them. Keeping
# the palette consistent with the UMAP was judged more important than local
# separation, since a reader moving between the two figures maps compartments by
# colour.
COMP_SLATE = {"Epithelial": "#748190", "Endothelial": "#e58677",
              "Stromal": "#367f80", "Immune": "#c58a4c"}
SHORT = {"Epithelial": "Epithelial", "Lymphatic endothelial": "Lymph. endo.",
         "Vascular endothelial": "Vasc. endo.", "Proliferating stromal": "Prolif. strom.",
         "Stromal (remodeling)": "Stromal (rem.)", "Smooth muscle": "Sm. muscle",
         "Fibroblast (Igf1+)": "Fib. Igf1+", "Dendritic cell": "DC",
         "Dendritic cell (cDC1)": "cDC1", "Macrophage/APC": "Mac/APC",
         "Fibroblast": "Fibroblast", "Mesothelial": "Mesothelial",
         "Pericyte": "Pericyte", "Macrophage": "Macrophage",
         "NK cell": "NK cell", "T cell": "T cell"}


def beeswarm(x, half_height, bin_w, rng):
    """Density-driven vertical jitter: wide where genes pile up, narrow in the tails.

    Bins x, then spreads each bin's points symmetrically about zero with a width
    proportional to sqrt(bin count), so the silhouette reads as a distribution
    rather than a uniform band.
    """
    xs = np.sort(x)
    ys = np.zeros_like(xs)
    edges = np.arange(xs.min() - 1e-9, xs.max() + bin_w, bin_w)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, len(edges) - 1)
    counts = np.bincount(idx, minlength=len(edges))
    cmax = max(counts.max(), 1)
    for b in np.flatnonzero(counts):
        sel = np.flatnonzero(idx == b)
        k = sel.size
        w = half_height * np.sqrt(counts[b] / cmax)
        if k == 1:
            ys[sel] = 0.0
        else:
            lin = np.linspace(-w, w, k)
            ys[sel] = lin[rng.permutation(k)]
    return xs, ys


def rec_pooled(nm, ct):
    """n beyond this type's own noise floor, from the pooled-estimate table."""
    r = PR_SUMMARY[(PR_SUMMARY.contrast == f"CTRL_vs_{nm}") & (PR_SUMMARY.ctype == ct)]
    return int(r.n_beyond_noise.iloc[0])


def build(cap_ns=250, cap_sig=400, cap_pooled=110, seed=11, figsize=(27.0, 15.5),
          fs_title=30, fs_lab=24, fs_small=20, s_point=5.5):
    rng = np.random.default_rng(seed)
    rows = ROW_ORDER
    nrow, ncol = len(rows), len(STG)
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrow, ncol * 2 + 4,
                          # label column width is MEASURED, not guessed: the widest
                          # label ("Stromal (rem.)") needs 2.30 in at fs_lab=26, so
                          # 2.85 units at this figure width. Too narrow and the labels
                          # overrun leftward into the compartment swatches.
                          width_ratios=[0.80, 0.42, 0.26, 2.85] + [6.6, 0.86] * ncol,
                          left=0.008, right=0.995, top=0.885, bottom=0.170,
                          wspace=0.0, hspace=0.16)
    # dendrogram spine, hierarchical clustering of CONTROL pseudobulk profiles,
    # so the tree is a homeostatic reference, not inferred from the contrasts plotted.
    axd = fig.add_subplot(gs[:, 0])
    for xs_, ys_ in zip(DENDRO["dcoord"], DENDRO["icoord"]):
        axd.plot(xs_, ys_, color="#666", lw=2.0, solid_joinstyle="miter", zorder=3)
    axd.set_ylim(10 * nrow, 0)
    hmax = max(max(c) for c in DENDRO["dcoord"])
    axd.set_xlim(hmax * 1.04, -hmax * 0.22)     # leaf tips stop well short of the swatches
    axd.axis("off")
    lim = 11.0
    n_markers = 0
    for ri, ct in enumerate(rows):
        axc = fig.add_subplot(gs[ri, 2])
        axc.add_patch(plt.Rectangle((0.12, 0.18), 0.76, 0.64,
                                    color=COMP_SLATE[COMP[ct]], lw=0))
        axc.set_xlim(0, 1); axc.set_ylim(0, 1); axc.axis("off")
        axl = fig.add_subplot(gs[ri, 3])
        axl.text(0.98, 0.5, SHORT[ct], ha="right", va="center", fontsize=fs_lab)
        axl.set_xlim(0, 1); axl.axis("off")
        for ci, (nm, st) in enumerate(STG):
            ax = fig.add_subplot(gs[ri, 4 + ci * 2])
            axg = fig.add_subplot(gs[ri, 5 + ci * 2])
            axg.axis("off")
            rec = sm[(sm.contrast == f"CTRL_vs_{nm}") & (sm.ctype == ct)].iloc[0]
            ax.axvline(0, lw=1.0, color="#c8c8c8", zorder=0)
            if rec["mode"] == "not_testable" and (nm, ct) in pooled:
                # too few replicates for a test: pooled effect-size estimate against
                # this type's own sampling-noise floor (grey band), open circles.
                pd_ = pooled[(nm, ct)]
                thr = np.maximum(1.0, pd_.floor95.values)
                band = float(np.median(pd_.floor95.values))
                ax.add_patch(plt.Rectangle((-band, -1.15), 2 * band, 2.3,
                                           color="#ededed", lw=0, zorder=0))
                v = pd_.log2FC.values
                beyond = np.abs(v) >= thr
                # open circles cost ~4x the path data of filled dots, so the pooled
                # rows carry their own tighter cap
                for mask, col, cap, z in [(~beyond, NS_GREY, cap_pooled, 1),
                                          (beyond & (v < 0), CTRL_GREY, cap_pooled, 2),
                                          (beyond & (v > 0), PAL_STAGE[nm], cap_pooled, 3)]:
                    w = v[mask]
                    if w.size == 0:
                        continue
                    if w.size > cap:
                        w = w[rng.choice(w.size, cap, replace=False)]
                    w = np.clip(w, -lim * 0.99, lim * 0.99)
                    xx, yy = beeswarm(w, half_height=0.94, bin_w=0.34, rng=rng)
                    ax.scatter(xx, yy, s=s_point * 2.6, facecolors="none", edgecolors=col,
                               linewidths=0.65, zorder=z)
                axg.text(0.42, 0.5, f"{int(rec_pooled(nm, ct)):,}", fontsize=fs_lab,
                         ha="right", va="center", style="italic")
            elif rec["mode"] == "not_testable":
                ax.add_patch(plt.Rectangle((-lim, -1), 2 * lim, 2, color="#f2f2f2", lw=0, zorder=0))
                ax.text(0.5, 0.5, "not estimable", fontsize=fs_small - 2, color="#999",
                        ha="center", va="center", style="italic", transform=ax.transAxes)
                axg.text(0.5, 0.5, "—", fontsize=fs_lab, color="#999", ha="center", va="center")
            else:
                d = swarm[(nm, ct)]
                sig = ((d.q < 0.05) & (d.log2FC.abs() >= 1)).values
                lfc = d.log2FC.values
                for mask, col, cap, z in [(~sig, NS_GREY, cap_ns, 1),
                                          (sig & (lfc < 0), CTRL_GREY, cap_sig, 2),
                                          (sig & (lfc > 0), PAL_STAGE[nm], cap_sig, 3)]:
                    v = lfc[mask]
                    if v.size == 0:
                        continue
                    if v.size > cap:
                        v = v[rng.choice(v.size, cap, replace=False)]
                    v = np.clip(v, -lim * 0.99, lim * 0.99)
                    xx, yy = beeswarm(v, half_height=0.94, bin_w=0.34, rng=rng)
                    ax.scatter(xx, yy, s=s_point, c=col, lw=0, zorder=z)
                    n_markers += v.size
                axg.text(0.42, 0.5, f"{int(rec.n_sig):,}", fontsize=fs_lab,
                         ha="right", va="center")
            ax.set_xlim(-lim, lim); ax.set_ylim(-1.15, 1.15)
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            if ri == nrow - 1:
                ax.set_xticks([-8, 0, 8]); ax.tick_params(labelsize=fs_small)
                ax.set_xlabel("log$_2$FC", fontsize=fs_lab, labelpad=6)
            else:
                ax.set_xticks([]); ax.spines["bottom"].set_visible(False)
            if ri == 0:
                ax.set_title(f"Control vs {nm}", color=PAL_STAGE[nm],
                             fontsize=fs_title, pad=26)
                axg.text(0.42, 2.55, "n", fontsize=fs_lab, style="italic",
                         ha="right", va="center", transform=axg.transAxes)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    h1 = [Line2D([], [], marker="o", ls="", ms=13, color=PAL_STAGE[nn], label=f"up at {nn}")
          for nn, _ in STG]
    h2 = [Line2D([], [], marker="o", ls="", ms=13, color=CTRL_GREY, label="higher in control"),
          Line2D([], [], marker="o", ls="", ms=13, color=NS_GREY, label="not significant"),
          Line2D([], [], marker="o", ls="", ms=13, mfc="none", mec="#555", mew=1.2,
                 label="open = pooled estimate (too few replicates to test)"),
          Patch(color="#ededed", label="grey band = that type's sampling-noise floor")]
    h3 = [Patch(color=COMP_SLATE[c], label=c) for c in ["Epithelial", "Endothelial", "Stromal", "Immune"]]
    fig.legend(handles=h1 + h2[:2], loc="lower left", bbox_to_anchor=(0.052, 0.070),
               ncol=6, frameon=False, fontsize=fs_lab, handletextpad=0.4, columnspacing=1.7)
    fig.legend(handles=h2[2:], loc="lower left", bbox_to_anchor=(0.052, 0.038),
               ncol=2, frameon=False, fontsize=fs_lab, handletextpad=0.4, columnspacing=1.7)
    fig.legend(handles=h3, loc="lower left", bbox_to_anchor=(0.052, 0.006),
               ncol=4, frameon=False, fontsize=fs_lab, handletextpad=0.5, columnspacing=1.7)
    fig.text(0.043, 0.0215, "compartment", fontsize=fs_lab, ha="right", va="center", color="#555")
    fig.text(0.008, 0.988, "Each stage engages a different set of cell types",
             fontsize=fs_title + 2, ha="left", va="top")
    fig.text(0.008, 0.113,
             "n = significant genes (|log$_2$FC| $\\geq$ 1, q < 0.05) from all tested genes; "
             "italic n = genes beyond that type's own noise floor (pooled, not a p-value; do not rank types by it)  ·  "
             f"dots are a fixed-seed display subsample (max {cap_ns} non-significant, {cap_sig} per significant side)",
             fontsize=fs_small, color="#777", ha="left")
    return fig, n_markers
