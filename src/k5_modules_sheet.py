"""Per-compartment k = 5 pseudobulk modules with the New Homeostasis GO terms (Fig. 4 B-E).

    build(repo_root, out_dir) -> {panel_label: svg_path}   one A4 sheet

Top row     one heatmap per compartment: the k = 5 pseudobulk temporal modules, 1500
            genes x 5 stages. The module side bar is coloured by the stage the module
            peaks in, with the module id and gene count beside it ("M1 / n=392").
            The top two genes per module by ANOVA F are labelled with leader lines.
Bottom row  four independent GO-BP dotplots, one per compartment, of that compartment's
            New Homeostasis module (its k = 5 module peaking at PPD30). Each panel has
            its own term list and gene-ratio axis; the four do not form a shared-term
            matrix. Enrichment ran against all genes (Enrichr GO_Biological_Process_2025
            background, BH-adjusted).

These modules are clustered per compartment and are not the five whole-tissue
programs of Fig. 3. They are keyed M1-M5 by their own clustering and take the
viridis stage sample rather than the paper's stage palette. M5 is each compartment's
New Homeostasis module and is the only label shared with the rest of the paper.

Geometry: the sheet is exactly A4 (210 x 297 mm); the content occupies about 262 mm and
is not stretched. Every rectangle is solved in millimetres and placed with add_axes, so
type size and panel size are independent. SC is the single type multiplier (1.0 = the
supplementary deck ladder, 1.5 = this sheet); the figure is saved at its true physical
size, so fontsize N is N pt on the page. Place the SVG at 100 % in Illustrator.
"""
import os
import re
import textwrap

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

import render_svg

A4_W, A4_H = 210 / 25.4, 297 / 25.4
COMPS = ["epithelial", "stromal", "immune", "endothelial"]
CLAB = {"epithelial": "Epithelial", "stromal": "Stromal (fibroblast)",
        "immune": "Immune", "endothelial": "Endothelial"}
STAGES = ["z_CTRL", "z_E16_5", "z_PPD1", "z_PPD7", "z_PPD30"]
SLAB = ["Control", "E16.5", "PPD1", "PPD7", "PPD30"]

# stage/module colours: the exact viridis sample used by the delivered S6 k=5 panels
SCOL = ["#471365", "#39558c", "#238a8d", "#3fbc73", "#bddf26"]

# Compartment palette sampled off the LIANA circle-plot figure (dominant solid arc
# colours). Used for compartment title text in a-d and for the tinted row band plus
# title text in e-h. Its ROLE differs from SCOL: no axes carries both, so the teal
# Stromal #338181 never sits beside the teal PPD1 #238a8d. These values sit a few units
# off the paper's compartment palette (Epithelial #748190, Stromal #367f80, Endothelial
# #e58677, Immune #c58a4c); the shipped sheet's own sampling is kept so this figure
# matches the delivered deck.
CCOL = {"epithelial": "#768292", "stromal": "#338181",
        "immune": "#c68c4d", "endothelial": "#e68677"}
GREY = "#555555"
SIL = {"epithelial": 0.331, "stromal": 0.308, "immune": 0.340, "endothelial": 0.272}
NSAMP = {"epithelial": 15, "stromal": 15, "immune": 14, "endothelial": 14}

SC = 1.5                                   # type multiplier
FS_BASE = 8 * SC                           # panel titles
FS_ANN = 7 * SC                            # axis labels, sub-panel titles
FS_TICK = 6 * SC                           # tick labels, keys
FS_SMALL = 5.4 * SC                        # per-panel provenance line
FS_MOD = 5.6 * SC                          # module id + gene count
FS_GENE = 5.0 * SC                         # gene callouts

# Millimetre layout.
LM, RM, GAP = 8.0, 8.0, 4.0
SLOT = (210 - LM - RM - 3 * GAP) / 4       # 45.5 mm per column
MODL, BARW, BARGAP, GENEL = 11.5, 1.6, 1.0, 13.5
PW = SLOT - MODL - BARW - BARGAP - GENEL - 1.0
HM_TOP, HM_H = 276.0, 64.0                 # heatmap block
TERML, GPW = 27.0, 16.0
GO_TOP, GO_H = 126.0, 84.0

# Module sizes printed in the deck's panels a-d. Asserted, never used to select or
# reorder rows: if the staged table disagrees the build stops instead of drawing a
# figure that no longer matches the deck.
DECK_MODULE_SIZES = {
    "epithelial": {1: 379, 2: 329, 3: 197, 4: 282, 5: 313},
    "stromal": {1: 456, 2: 120, 3: 224, 4: 513, 5: 187},
    "immune": {1: 301, 2: 308, 3: 508, 4: 160, 5: 223},
    "endothelial": {1: 392, 2: 181, 3: 242, 4: 402, 5: 283},
}
# Significant-term counts printed in the deck's e-h subtitles, out of the 10 plotted.
DECK_GO_SIG = {"epithelial": 10, "stromal": 0, "immune": 10, "endothelial": 1}

# Contractions so no GO label exceeds three lines of the term column. Each keeps the
# biological reading; nothing is dropped, only contracted (deck convention).
ABBR = [
    ("Positive regulation of vascular endothelial growth factor production",
     "Pos. reg. VEGF production"),
    ("Regulation of vascular endothelial growth factor production",
     "Reg. VEGF production"),
    ("Cell-cell adhesion via plasma-membrane adhesion molecules",
     "Cell-cell adhesion, membrane molecules"),
    ("Negative regulation of lamellipodium organization",
     "Neg. reg. lamellipodium organization"),
    ("Negative regulation of RNA biosynthetic process",
     "Neg. reg. RNA biosynthesis"),
    ("Regulation of blood-brain barrier permeability",
     "Reg. blood-brain barrier permeability"),
    ("Regulation of calcium ion-dependent exocytosis",
     "Reg. calcium-dependent exocytosis"),
    ("Calcium ion transmembrane import into cytosol",
     "Calcium import into cytosol"),
    ("Positive regulation of vascular permeability",
     "Pos. reg. vascular permeability"),
    ("Positive regulation of programmed cell death",
     "Pos. reg. programmed cell death"),
    ("Cellular response to growth factor stimulus",
     "Response to growth factor stimulus"),
    ("Positive regulation of leukocyte chemotaxis",
     "Pos. reg. leukocyte chemotaxis"),
    ("Positive regulation of response to wounding",
     "Pos. reg. response to wounding"),
    ("Negative regulation of leukocyte migration",
     "Neg. reg. leukocyte migration"),
    ("Positive regulation of apoptotic process",
     "Pos. reg. apoptotic process"),
    ("Regulation of epithelial cell migration",
     "Reg. epithelial cell migration"),
    ("Regulation of protein catabolic process", "Reg. protein catabolism"),
    ("Positive regulation of catabolic process", "Pos. reg. catabolism"),
    ("Positive regulation of cell motility", "Pos. reg. cell motility"),
    ("Epidermal cell differentiation", "Epidermal cell differentiation"),
    ("Cellular response to decreased oxygen levels",
     "Response to decreased oxygen"),
    ("Regulation of cardiac muscle hypertrophy",
     "Reg. cardiac muscle hypertrophy"),
    ("Positive regulation of organelle organization",
     "Pos. reg. organelle organization"),
    ("Cellular response to hypoxia", "Response to hypoxia"),
    ("Inorganic anion transport", "Inorganic anion transport"),
    ("Regulation of angiogenesis", "Reg. angiogenesis"),
    ("Positive regulation of cell adhesion", "Pos. reg. cell adhesion"),
]
ABBR.sort(key=lambda p: -len(p[0]))


def _fig5_k5(repo_root):
    return os.path.join(repo_root, "data", "raw_light", "fig5_k5")


def load_inputs(repo_root):
    """Read the four staged k=5 tables and resolve M5 per compartment.

    Returns (K, sets, GSEL, audit):
      K {comp: DataFrame} the 1500 x 5 z-scored module table, one per compartment
      sets {comp: {"module": int}} which module id is that compartment's M5
      GSEL {comp: DataFrame} the 10 plotted GO rows, gene_ratio renamed to ratio
      audit {"module_sizes", "go_sig", "m5_module", "n_terms"} for the caller to report

    M5 is resolved as the module whose mean z peaks at PPD30, not by taking id 5. The
    ids happen to agree in all four compartments, and that agreement is asserted rather
    than assumed. Immune has two modules peaking at PPD7 (M3 and M4), so an index-based
    rule would be wrong the moment the clustering is rerun.
    """
    d = _fig5_k5(repo_root)
    need = {"modules": ("Fig5_k5_modules_source.csv", 6000),
            "m5_genes": ("Fig5_NewHomeostasis_M5_genes.csv", 1006),
            "go_plotted": ("Fig5_NewHomeostasis_GO_plotted.csv", 40),
            "go_allterms": ("Fig5_NewHomeostasis_GO_allterms.csv", 6348)}
    tab = {}
    for key, (fn, n_exp) in need.items():
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"missing staged input {fn}. Expected at {p}; it is recorded in "
                f"data/raw_light/MANIFEST.csv as fig5_k5/{fn}")
        t = pd.read_csv(p)
        if len(t) != n_exp:
            raise ValueError(f"{fn}: expected {n_exp} rows, got {len(t)}")
        tab[key] = t

    mods, go = tab["modules"], tab["go_plotted"]
    inv = {v: k for k, v in CLAB.items()}
    for name, t in (("modules_source", mods), ("GO_plotted", go)):
        unknown = set(t.compartment.unique()) - set(inv)
        if unknown:
            raise ValueError(f"{name}: unrecognised compartment label(s) {sorted(unknown)}")
    mods = mods.assign(comp=mods.compartment.map(inv))
    go = go.assign(comp=go.compartment.map(inv))

    K, sets, GSEL = {}, {}, {}
    module_sizes, go_sig, m5_module, n_terms = {}, {}, {}, {}
    for c in COMPS:
        d_c = mods[mods.comp == c].copy()
        if len(d_c) != 1500:
            raise ValueError(f"{c}: expected 1500 genes, got {len(d_c)}")
        sizes = d_c.groupby("module").size().to_dict()
        module_sizes[c] = {int(k): int(v) for k, v in sizes.items()}
        if module_sizes[c] != DECK_MODULE_SIZES[c]:
            raise ValueError(f"{c} module sizes {module_sizes[c]} disagree with the deck "
                             f"{DECK_MODULE_SIZES[c]}")
        peaks = {int(m): d_c[d_c.module == m][STAGES].mean().idxmax()
                 for m in sorted(d_c.module.unique())}
        at_ppd30 = [m for m, p in peaks.items() if p == "z_PPD30"]
        if len(at_ppd30) != 1:
            raise ValueError(f"{c}: expected exactly one module peaking at PPD30, "
                             f"got {at_ppd30} from peaks {peaks}")
        m5 = at_ppd30[0]
        m5_module[c] = m5
        K[c] = d_c
        sets[c] = {"module": m5}

        g = go[go.comp == c].copy()
        if len(g) != 10:
            raise ValueError(f"{c}: expected 10 plotted GO terms, got {len(g)}")
        if set(g.module) != {"M5"}:
            raise ValueError(f"{c}: GO_plotted carries module(s) {set(g.module)}, "
                             f"expected only M5")
        if set(g.module_size) != {module_sizes[c][m5]}:
            raise ValueError(f"{c}: GO module_size {set(g.module_size)} disagrees with "
                             f"the M{m5} gene count {module_sizes[c][m5]}")
        GSEL[c] = g.rename(columns={"gene_ratio": "ratio"})
        go_sig[c] = int((g.padj < 0.05).sum())
        if go_sig[c] != DECK_GO_SIG[c]:
            raise ValueError(f"{c}: {go_sig[c]}/10 terms at q<0.05 disagrees with the "
                             f"deck subtitle {DECK_GO_SIG[c]}/10")

    # M5 gene counts must agree between the standalone M5 table and module M5 of the
    # source table. The M5 table is not plotted; it is the cross-check on that split.
    m5t = tab["m5_genes"].assign(comp=tab["m5_genes"].compartment.map(inv))
    for c in COMPS:
        n_t = int((m5t.comp == c).sum())
        n_k = module_sizes[c][m5_module[c]]
        if n_t != n_k:
            raise ValueError(f"{c}: M5 gene table has {n_t} genes, module "
                             f"M{m5_module[c]} of the source table has {n_k}")

    # The e-h header prints how many terms enrichment returned. Read it off the full
    # output instead of hardcoding, so the header cannot drift from the table.
    allt = tab["go_allterms"].assign(comp=tab["go_allterms"].compartment.map(inv))
    n_terms = {c: int((allt.comp == c).sum()) for c in COMPS}

    audit = dict(module_sizes=module_sizes, go_sig=go_sig, m5_module=m5_module,
                 n_terms=n_terms)
    return K, sets, GSEL, audit


def tidy_term(t):
    """'Inflammatory Response (GO:0006954)' -> 'Inflammatory response', contracted."""
    m = re.match(r"^(.*?)\s*\(GO:\d+\)\s*$", t)
    name = m.group(1) if m else t
    name = name[0].upper() + name[1:].lower()
    for w in ["rna", "dna", "vegf"]:
        name = re.sub(rf"\b{w}\b", w.upper(), name)
    for long_, short_ in ABBR:
        name = name.replace(long_, short_)
    return name


def _rect(x, y, w, h):
    """mm rectangle (y = BOTTOM edge, mm from page bottom) -> figure fractions."""
    return [x / 210.0, y / 297.0, w / 210.0, h / 297.0]


def _heatmap(ax, d, star_mod, nlab=2):
    """One compartment's k=5 module heatmap.

    The module bar, its "Mk / n=" label and the gene callouts are all drawn in a
    blended (axes-x, data-y) transform, so they scale with the panel and cannot spill
    into the neighbouring column. Genes are ordered within a module by descending z at
    that module's peak stage.
    """
    Z, mods, sizes, peaks, labs = [], [], [], [], []
    row = 0
    for m in sorted(d.module.unique()):
        sub = d[d.module == m]
        peak = sub[STAGES].mean().idxmax()
        sub = sub.sort_values(peak, ascending=False).reset_index(drop=True)
        Z.append(sub[STAGES].values)
        for gi, g in sub.nlargest(nlab, "F").iterrows():
            labs.append((row + gi + 0.5, g.gene, STAGES.index(peak)))
        mods.append(m); sizes.append(len(sub)); peaks.append(STAGES.index(peak))
        row += len(sub)
    Zm = np.vstack(Z)
    N = len(Zm)
    vmax = np.percentile(np.abs(Zm), 99)
    ax.imshow(Zm, aspect="auto", cmap="RdBu_r",
              norm=TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax),
              interpolation="nearest", extent=[-0.5, 4.5, N, 0])
    tr = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)

    bar_r = -BARGAP / PW                      # bar right edge, axes fraction
    bar_l = -(BARGAP + BARW) / PW
    y0 = 0
    for m, n, pi in zip(mods, sizes, peaks):
        ax.add_patch(mpl.patches.Rectangle(
            (bar_l, y0), BARW / PW, n, transform=tr, color=SCOL[pi], lw=0,
            clip_on=False, zorder=5))
        star = "*" if m == star_mod else ""
        ax.text(bar_l - 0.055, y0 + n / 2, f"M{m}{star}\nn={n}", transform=tr,
                ha="right", va="center", fontsize=FS_MOD, color=SCOL[pi],
                linespacing=1.2, zorder=6)
        y0 += n

    # gene callouts: de-collide along y against a minimum pitch, then draw the leader
    labs.sort()
    pitch = N * 0.055
    ys = [l[0] for l in labs]
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < pitch:
            ys[i] = ys[i - 1] + pitch
    shift = max(0.0, ys[-1] - (N - pitch * 0.4))
    ys = [y - shift for y in ys]
    for (y_src, gene, pi), y_dst in zip(labs, ys):
        ax.plot([1.005, 1 + 0.6 / PW], [y_src, y_dst], transform=tr, lw=0.4,
                color="#9a9a9a", clip_on=False, zorder=5, solid_capstyle="butt")
        ax.text(1 + 1.0 / PW, y_dst, gene, transform=tr, ha="left", va="center",
                fontsize=FS_GENE, color=SCOL[pi], style="italic", zorder=6)

    ax.set_xlim(-0.5, 4.5); ax.set_ylim(N, 0)
    ax.set_yticks([])
    ax.set_xticks(range(5))
    ax.set_xticklabels(SLAB, rotation=90, fontsize=FS_TICK)
    for t, sc in zip(ax.get_xticklabels(), SCOL):
        t.set_color(sc)
    ax.tick_params(axis="x", length=2, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(True); s.set_linewidth(0.6); s.set_color("black")


def _godot(ax, sel, band_col, dot_scale, wrap=20):
    """One compartment's independent GO dotplot: own terms, own gene-ratio axis.

    Deck idiom: dot area = genes in term, colour = -log10 adjusted P (viridis), rows in
    a band tinted with the module's peak-stage colour, x-grid only.
    """
    sel = sel.sort_values("ratio", ascending=False).reset_index(drop=True)
    n = len(sel)
    y = np.arange(n)
    ax.axhspan(-0.5, n - 0.5, color=band_col, alpha=0.13, lw=0, zorder=0)
    nlp = -np.log10(sel.padj.clip(lower=1e-300))
    sc = ax.scatter(sel.ratio, y, s=(4.2 * sel.k.values + 18) * dot_scale,
                    c=nlp, cmap="viridis", vmin=0, vmax=nlp.max(),
                    edgecolors="none", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels([textwrap.fill(tidy_term(t), wrap) for t in sel.term],
                       fontsize=FS_SMALL, linespacing=1.05)
    ax.set_ylim(n - 0.5, -0.5)
    lo, hi = sel.ratio.min(), sel.ratio.max()
    pad = 0.18 * (hi - lo)
    ax.set_xlim(max(0, lo - pad), hi + pad)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(3))
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(
        lambda v, _: f"{v:.02f}".lstrip("0")))
    ax.tick_params(axis="x", labelsize=FS_TICK, length=2, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)
    for nm in ("top", "right"):
        ax.spines[nm].set_visible(False)
    for nm in ("left", "bottom"):
        ax.spines[nm].set_visible(True); ax.spines[nm].set_linewidth(0.6)
    ax.set_axisbelow(True)
    ax.grid(axis="x", color="#e8e8e8", lw=0.5, zorder=1)
    return sc


def _sheet(K, sets, GSEL, audit):
    """Draw the whole A4 sheet. Returns the figure."""
    fig = plt.figure(figsize=(A4_W, A4_H))
    DS = 0.62 * SC ** 1.4                   # dot area scales with type, not with rows

    # Panels a-d: k=5 module heatmaps.
    for j, c in enumerate(COMPS):
        x0 = LM + j * (SLOT + GAP) + MODL + BARW + BARGAP
        ax = fig.add_axes(_rect(x0, HM_TOP - HM_H, PW, HM_H))
        _heatmap(ax, K[c], sets[c]["module"])
        xl = LM + j * (SLOT + GAP)
        fig.text(xl / 210, (HM_TOP + 14.0) / 297, chr(97 + j), fontsize=FS_BASE,
                 fontweight="bold", ha="left", va="baseline")
        fig.text(xl / 210, (HM_TOP + 8.5) / 297, CLAB[c], fontsize=FS_BASE,
                 color=CCOL[c], fontweight="bold", ha="left", va="baseline")
        fig.text(xl / 210, (HM_TOP + 3.5) / 297,
                 f"k=5 \u00b7 sil {SIL[c]:.2f} \u00b7 n={NSAMP[c]}",
                 fontsize=FS_SMALL, color=GREY, ha="left", va="baseline")

    # z colourbar (left) and stage key (right) share one band under the heatmaps.
    ZY = 191.0
    cax = fig.add_axes(_rect(LM + MODL + 4, ZY, 30, 1.8))
    cb = fig.colorbar(mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(-1.8, 1.8), cmap="RdBu_r"),
        cax=cax, orientation="horizontal", ticks=[-1.5, 0, 1.5])
    cb.set_label("pseudobulk z-score (0-centred)", fontsize=FS_TICK, labelpad=2)
    cb.ax.tick_params(labelsize=FS_TICK, length=2, pad=1)
    cb.outline.set_linewidth(0.4)

    KYS = ZY - 12.0
    fig.text(LM / 210, (KYS - 7.0) / 297,
             "module side bar = the stage that module peaks in   \u00b7   "
             "M5* = the New Homeostasis module carried into e\u2013h",
             fontsize=FS_TICK, color=GREY, ha="left", va="baseline")
    kx = LM
    for i, (sc_, sl) in enumerate(zip(SCOL, SLAB)):
        x = kx + i * 23.0
        fig.patches.append(mpl.patches.Rectangle(
            _rect(x, KYS, 3.0, 2.2)[:2], 3.0 / 210, 2.2 / 297,
            transform=fig.transFigure, facecolor=sc_, lw=0, figure=fig))
        fig.text((x + 4.2) / 210, (KYS + 0.4) / 297, sl, fontsize=FS_TICK,
                 color=sc_, ha="left", va="baseline")

    # e-h header. The term-count range comes off the full enrichment output.
    lo_t, hi_t = min(audit["n_terms"].values()), max(audit["n_terms"].values())
    fig.text(LM / 210, (GO_TOP + 42.0) / 297,
             "New Homeostasis module (M5 \u2014 the k=5 module peaking at PPD30):\n"
             "unbiased GO biological process, run separately per compartment",
             fontsize=FS_BASE, ha="left", va="top", linespacing=1.35)
    fig.text(LM / 210, (GO_TOP + 30.0) / 297,
             "1500 genes tested per compartment. Each compartment's own M5 gene set is "
             "tested against ALL genes (Enrichr GO_BP 2025 background,\nBH-adjusted); "
             f"top 10 of the {lo_t:,}\u2013{hi_t:,} terms returned, ranked by adjusted "
             "P. The four term lists are independent \u2014 the panels\nshare neither "
             "rows nor a common axis.",
             fontsize=FS_SMALL, color=GREY, ha="left", va="top", linespacing=1.35)

    # Panels e-h: four independent GO dotplots.
    gax, scs = [], []
    for j, c in enumerate(COMPS):
        x0 = LM + j * (SLOT + GAP) + TERML
        ax = fig.add_axes(_rect(x0, GO_TOP - GO_H, GPW, GO_H))
        gax.append(ax)
        scs.append(_godot(ax, GSEL[c], CCOL[c], dot_scale=DS))
        xl = LM + j * (SLOT + GAP)
        fig.text(xl / 210, (GO_TOP + 14.0) / 297, chr(101 + j), fontsize=FS_BASE,
                 fontweight="bold", ha="left", va="baseline")
        fig.text(xl / 210, (GO_TOP + 7.0) / 297, CLAB[c], fontsize=FS_ANN,
                 color=CCOL[c], fontweight="bold", ha="left", va="baseline")
        nsig = audit["go_sig"][c]
        nmod = audit["module_sizes"][c][audit["m5_module"][c]]
        fig.text(xl / 210, (GO_TOP + 2.0) / 297,
                 f"M5 \u00b7 {nmod} genes \u00b7 {nsig}/10 q<0.05",
                 fontsize=FS_SMALL, color=GREY, ha="left", va="baseline")

    GB = GO_TOP - GO_H
    fig.text((LM + (210 - LM - RM) / 2) / 210, (GB - 10.0) / 297,
             "gene ratio (module genes in term / module size)",
             fontsize=FS_ANN, ha="center", va="baseline")

    # Per-panel colourbars: each GO panel has its own P range, so one shared bar would
    # misstate three of the four panels.
    for ax, sc in zip(gax, scs):
        p = ax.get_position()
        ca = fig.add_axes([p.x0, (GB - 22.0) / 297, p.width, 1.6 / 297])
        cbj = fig.colorbar(sc, cax=ca, orientation="horizontal")
        cbj.locator = mpl.ticker.MaxNLocator(3); cbj.update_ticks()
        cbj.ax.tick_params(labelsize=FS_TICK * 0.88, length=1.6, pad=1)
        cbj.outline.set_linewidth(0.35)
    # Plain string, not mathtext: matplotlib renders $...$ as one <tspan> PER GLYPH,
    # which is live text but tedious to re-edit. U+2212 minus plus a literal "10".
    fig.text(LM / 210, (GB - 17.5) / 297,
             "\u2212log10 adjusted P \u2014 own scale per panel",
             fontsize=FS_TICK, color=GREY, ha="left", va="baseline")

    # dot-size key
    KY = GB - 36.0
    kax = fig.add_axes(_rect(LM, KY, 46, 7.0))
    kax.set_axis_off(); kax.set_xlim(0, 1); kax.set_ylim(0, 1)
    ref = [3, 10, 17]
    for xp, r in zip([0.10, 0.42, 0.80], ref):
        kax.scatter([xp], [0.55], s=(4.2 * r + 18) * DS, c="#8a8a8a",
                    edgecolors="none", clip_on=False)
        kax.text(xp, 0.0, str(r), ha="center", va="baseline", fontsize=FS_TICK)
    kax.text(0.0, 1.02, "dot area = genes in term", fontsize=FS_TICK, color=GREY,
             ha="left", va="baseline")
    return fig


def build(repo_root: str, out_dir: str) -> dict:
    """Render the Fig5 A4 sheet into out_dir. Returns {panel_label: svg_path}.

    Panels a-h all live on one sheet, so every label maps to the same file, as Fig4 a-e
    already does.
    """
    os.makedirs(out_dir, exist_ok=True)
    K, sets, GSEL, audit = load_inputs(repo_root)
    render_svg.svg_mode(base=FS_BASE, small=FS_ANN, tiny=FS_TICK)
    fig = _sheet(K, sets, GSEL, audit)

    # save_svg saves with bbox_inches='tight', which crops the canvas to the drawn ink
    # and would land well short of A4 on a sheet that leaves 35 mm of page empty. A
    # zero-size invisible artist spanning the whole figure gives the tight bbox nothing
    # to crop, and pad_inches=0 stops it adding a margin back. Measured: 595.275591 x
    # 841.889764 pt = exactly 210 x 297 mm. The alternative, savefig.bbox='standard',
    # cannot be reached through save_svg without editing the helper.
    fig.add_artist(mpl.patches.Rectangle(
        (0, 0), 1, 1, transform=fig.transFigure, fill=False, ec="none", lw=0))
    stem = os.path.join(out_dir, "Fig4BE_k5_modules_GO_A4")
    with mpl.rc_context({"savefig.pad_inches": 0.0}):
        overlaps = render_svg.save_svg(fig, stem, also_pdf=True)
    plt.close(fig)

    info = render_svg.verify(stem + ".svg", expect_words=("New Homeostasis", "n=313"))
    if info["text_elements"] <= 0:
        raise ValueError(f"{stem}.svg has no live <text>: {info}")
    size_pt = _svg_size_pt(stem + ".svg")
    canvas_mm = tuple(round(v / 72 * 25.4, 3) for v in size_pt)
    if canvas_mm != (210.0, 297.0):
        raise ValueError(f"canvas is {canvas_mm} mm, expected (210.0, 297.0)")

    build.report = dict(module_sizes=audit["module_sizes"], go_sig=audit["go_sig"],
                        m5_module=audit["m5_module"], n_terms=audit["n_terms"],
                        canvas_mm=canvas_mm, verify=info, text_overlaps=len(overlaps))
    return {k: stem + ".svg" for k in "abcdefgh"}


def _svg_size_pt(path):
    """The saved SVG's declared width and height in points."""
    with open(path, encoding="utf-8") as fh:
        head = fh.read(2000)
    m = re.search(r'width="([\d.]+)pt"\s+height="([\d.]+)pt"', head)
    if not m:
        raise ValueError(f"no pt canvas size in the SVG header of {path}")
    return float(m.group(1)), float(m.group(2))


if __name__ == "__main__":
    import sys
    r = sys.argv[1]
    print(build(r, os.path.join(r, "figures")))
