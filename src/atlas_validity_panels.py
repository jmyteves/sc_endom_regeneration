"""Annotation validity and batch integration (fig. S2 E, G, H) and three extra panels.

    load_inputs(repo_root) -> dict of tables
    panel_d  Harmony UMAP coloured by cell identity and by stage           fig. S2 E
    panel_f  cells recovered per replicate, stacked by cell type           fig. S2 G
    panel_g  within-replicate composition                                  fig. S2 H
    panel_a  marker grid over the 16 labels (26-gene column set)           extra
    panel_e  QC covariates on the Harmony UMAP                             extra
    panel_h  equal-N subsampling against full depth                        extra

Not reproducible, returned as NOT COMPUTABLE with a placeholder slug:
  b, c      pseudobulk signature correlation against the MCA uterus reference and
            its compartment summary (fig. S2 F). T3_MCA_correlation_and_mapping.csv
            records MCA_pearson_r as not_recorded_on_disk; the 16 x 12 Pearson
            matrix exists only as a raster in the delivered figure. The cluster ->
            cell-type mapping in that file is real and panels f and g use it.
  d_rawPCA  the raw-PCA half of fig. S2 E. The un-integrated UMAP was never staged,
            and running UMAP on X_pca would produce a new embedding, not that one.

Panel a differs from the delivered raster in two ways, both reported rather than
patched: the delivered panel labels 17 rows including 'Myeloid (cl20)', a label
absent from every annotation table (cluster 20 is Smooth muscle); and 6 of its 26
genes are absent from the staged z-score matrix, so those columns are drawn as n.d.
No percent-expressing table exists, so dot area is not reproduced and the grid
encodes z-score in colour only.

Inputs
  data/staged/cells.parquet              698,631 cells, Harmony UMAP + QC
  data/staged/marker_zscore_deck139.csv  16 x 139 gene-wise z-scores
  data/raw_light/palette_final.json      cell-identity palette
  data/raw_light/sfig2/*.csv             the S2 source tables
"""
import json
import os

import matplotlib as mpl
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
COMPARTMENTS = ["Epithelial", "Stromal", "Endothelial", "Immune"]
COMPARTMENT_COLOR = {"Epithelial": "#748190", "Stromal": "#367f80",
                     "Endothelial": "#e58677", "Immune": "#c58a4c"}

TYPE_ORDER = [
    "Epithelial",
    "Fibroblast", "Mesothelial", "Proliferating stromal", "Stromal (remodeling)",
    "Pericyte", "Smooth muscle", "Fibroblast (Igf1+)",
    "Vascular endothelial", "Lymphatic endothelial",
    "Macrophage", "Dendritic cell (cDC1)", "NK cell", "Macrophage/APC", "T cell",
    "Dendritic cell",
]
TYPE_COMPARTMENT = {
    "Epithelial": "Epithelial",
    "Fibroblast": "Stromal", "Mesothelial": "Stromal",
    "Proliferating stromal": "Stromal", "Stromal (remodeling)": "Stromal",
    "Pericyte": "Stromal", "Smooth muscle": "Stromal",
    "Fibroblast (Igf1+)": "Stromal",
    "Vascular endothelial": "Endothelial", "Lymphatic endothelial": "Endothelial",
    "Macrophage": "Immune", "Dendritic cell (cDC1)": "Immune", "NK cell": "Immune",
    "Macrophage/APC": "Immune", "T cell": "Immune", "Dendritic cell": "Immune",
}
TYPE_SHORT = {
    "Proliferating stromal": "Prolif. stromal", "Stromal (remodeling)": "Stromal (remod.)",
    "Smooth muscle": "Sm. muscle", "Fibroblast (Igf1+)": "Fib. Igf1+",
    "Vascular endothelial": "Vasc. endo.", "Lymphatic endothelial": "Lymph. endo.",
    "Dendritic cell (cDC1)": "cDC1", "Macrophage/APC": "Mac/APC",
    "Dendritic cell": "DC",
}

# The 26 gene columns of the delivered S2a panel, in its order. Read from the
# live <text> nodes of figures_svg/S2a_marker_dotplot_17types.svg and recorded in
# SFigs_latest/docs/_manifest_S2.csv, so the order is the rendered one.
DECK26 = ["Dpt", "Dcn", "Col1a1", "Rxfp1", "Mmp11", "Cd74", "H2-Aa", "Lyz2",
          "Csf1r", "Acta2", "Myh11", "Pecam1", "Cdh5", "Lyve1", "Prox1", "Epcam",
          "Krt8", "Krt18", "Mki67", "Top2a", "Cd3e", "Nkg7", "Klrb1c", "Msln",
          "Upk3b", "Pdgfra"]
# Cluster 20 carries 'Myeloid (cl20)' in the delivered panel a, panel d title and
# panel e cluster list. Every current table calls it Smooth muscle. The label is
# kept here only so build() can report the mismatch.
RETIRED_LABEL = "Myeloid (cl20)"
RETIRED_LABEL_IS_NOW = "Smooth muscle"

NOT_COMPUTABLE = {
    "b": "pseudobulk signature correlation of the 16 control cell types against "
         "the 12 MCA uterus types: T3 records MCA_pearson_r as "
         "not_recorded_on_disk and the 16x12 matrix survives only as an "
         "embedded raster in the delivered SVG",
    "c": "compartment-level concordance against the MCA reference: same source "
         "gap as panel b, only the delivered raster carries the values",
    "d_rawPCA": "raw-PCA UMAP half of panel d: X_pca is staged but the "
                "un-integrated UMAP is not, and re-running UMAP would make a "
                "new embedding rather than reproduce the delivered one",
}
# cells.parquet carries n_counts and n_genes per cell. Mitochondrial percent and
# the doublet score are per-cell quantities that exist only inside the h5ad, and
# opening it is scripts/ work, so panel e shows two of the delivered four.
QC_ABSENT = "mitochondrial % and doublet score are not staged per cell"


def _p(repo_root, *parts):
    return os.path.join(repo_root, "data", *parts)


def _need(path, hint):
    if not os.path.exists(path):
        raise FileNotFoundError(f"SFig2 input missing: {path} ({hint})")
    return path


def resolve_tombstone(path):
    """Follow a superseded source table to the file that replaced it.

    Four S2g_* tables were reduced to a single row of (status, superseded_by,
    reason) when the equal-N content moved to panel h. Reading the pointer keeps
    the module working off whichever file is current.
    """
    df = pd.read_csv(path)
    if list(df.columns) == ["status", "superseded_by", "reason"]:
        return os.path.join(os.path.dirname(path), df["superseded_by"].iloc[0])
    return path


def load_inputs(repo_root):
    """Every table SFig2 reads, with tombstones already followed."""
    sfig2 = _p(repo_root, "raw_light", "sfig2")
    comp_path = _need(os.path.join(sfig2, "S2g_composition_per_replicate_source.csv"),
                      "copy from SFigs_latest/tables/")
    stats_path = resolve_tombstone(
        _need(os.path.join(sfig2, "S2g_statistics_source.csv"),
              "copy from SFigs_latest/tables/"))
    eq_path = resolve_tombstone(
        _need(os.path.join(sfig2, "S2g_equalN_vs_fulldepth_source.csv"),
              "copy from SFigs_latest/tables/"))
    boot_path = resolve_tombstone(
        _need(os.path.join(sfig2, "S2g_equalN_bootstrap_shares_source.csv"),
              "copy from SFigs_latest/tables/"))
    mca_path = _need(os.path.join(sfig2, "T3_MCA_correlation_and_mapping.csv"),
                     "copy from SFigs_latest/tables_supplementary/")

    comp = pd.read_csv(comp_path)
    counts = comp[comp["panel"] == "S2g_a_absolute_counts"].copy()
    pct = comp[comp["panel"] == "S2g_b_within_replicate_percent"].copy()
    if len(counts) != 14 or len(pct) != 14:
        raise ValueError(f"S2g composition should hold 14 rows per sub-panel, "
                         f"got {len(counts)} and {len(pct)}")

    stats = pd.read_csv(stats_path).set_index("statistic")
    equal_n = pd.read_csv(eq_path)
    boot = pd.read_csv(boot_path)
    mca = pd.read_csv(mca_path)

    cells = pd.read_parquet(_need(_p(repo_root, "staged", "cells.parquet"),
                                  "written by scripts/00_stage_inputs.py"))
    mz = pd.read_csv(_need(_p(repo_root, "staged", "marker_zscore_deck139.csv"),
                           "written by scripts/stage_marker_deck139.py"),
                     index_col=0)
    pal = json.load(open(_need(_p(repo_root, "raw_light", "palette_final.json"),
                               "cell-identity palette")))
    missing = [t for t in TYPE_ORDER if t not in pal]
    if missing:
        raise ValueError(f"palette_final.json lacks colour(s) for: {missing}")
    if list(mz.index.sort_values()) != sorted(TYPE_ORDER):
        raise ValueError("marker_zscore_deck139.csv rows are not the 16 labels")
    return {"counts": counts, "pct": pct, "stats": stats, "equal_n": equal_n,
            "boot": boot, "mca": mca, "cells": cells, "mz": mz,
            "palette": {t: pal[t] for t in TYPE_ORDER}}


def deck26_matrix(mz):
    """16 x 26 z-scores on the deck's gene columns, absent genes left as NaN."""
    present = [g for g in DECK26 if g in mz.columns]
    absent = [g for g in DECK26 if g not in mz.columns]
    m = mz.reindex(index=TYPE_ORDER, columns=DECK26)
    return m, present, absent


def _umap_scatter(ax, df, colour, title, s=0.06):
    ax.scatter(df["umap_x"], df["umap_y"], s=s, c=colour, linewidths=0,
               rasterized=True)
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=7.5)


def _shuffled(cells, seed=0):
    """Draw order shuffled so the last stage plotted does not sit on top."""
    rng = np.random.default_rng(seed)
    return cells.iloc[rng.permutation(len(cells))]


def panel_a(mz, out_dir):
    """Marker grid over the 16 labels on the deck's 26 gene columns."""
    m, present, absent = deck26_matrix(mz)
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    im = ax.imshow(np.ma.masked_invalid(m.values), aspect="auto", cmap="viridis",
                   vmin=-2, vmax=2, interpolation="nearest")
    for j, g in enumerate(DECK26):
        if g in absent:
            ax.add_patch(Rectangle((j - .5, -.5), 1, len(TYPE_ORDER),
                                   facecolor="#f2f2f2", edgecolor="none", zorder=2))
            ax.text(j, (len(TYPE_ORDER) - 1) / 2, "n.d.", ha="center", va="center",
                    fontsize=4.6, rotation=90, color="#a03020", zorder=3)
    ax.set_yticks(range(len(TYPE_ORDER)))
    ax.set_yticklabels([TYPE_SHORT.get(t, t) for t in TYPE_ORDER], fontsize=5.6)
    ax.set_xticks(range(len(DECK26)))
    ax.set_xticklabels(DECK26, rotation=90, fontsize=5.2, style="italic")
    ax.tick_params(length=1.2, pad=1.0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # Compartment brackets on the left edge, so a reader can see the block
    # structure without reading all 16 labels.
    start = 0
    for c in COMPARTMENTS:
        n = sum(1 for t in TYPE_ORDER if TYPE_COMPARTMENT[t] == c)
        ax.plot([-3.4, -3.4], [start - 0.35, start + n - 0.65], lw=2.4,
                color=COMPARTMENT_COLOR[c], clip_on=False, zorder=4)
        start += n
    ax.set_title("Cell-identity markers, gene-wise z-score", fontsize=8)
    cb = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.015, ticks=[-2, -1, 0, 1, 2])
    cb.set_label("Z-score", fontsize=6)
    cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_visible(False)
    fig.text(0.02, 0.015, f"{len(present)}/26 deck genes are in the staged "
             f"matrix; {', '.join(absent)} are not, and are drawn n.d. "
             f"Colour only: no percent-expressing table exists, so dot area is "
             f"not reproducible.", fontsize=5.0, ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    stem = os.path.join(out_dir, "extra_marker_grid")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_d(cells, palette, out_dir):
    """Harmony UMAP by cell identity and by stage."""
    d = _shuffled(cells)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    _umap_scatter(axes[0], d, [palette[t] for t in d["celltype"]],
                  "Harmony UMAP by cell identity")
    axes[0].legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.2,
                                   mfc=palette[t], mec="none",
                                   label=TYPE_SHORT.get(t, t)) for t in TYPE_ORDER],
                   fontsize=4.6, loc="center left", bbox_to_anchor=(1.0, 0.5),
                   ncol=1, handletextpad=0.3, labelspacing=0.25)
    _umap_scatter(axes[1], d, [STAGE_COLOR[s] for s in d["stage"].astype(str)],
                  "Harmony UMAP by stage")
    axes[1].legend(handles=[Line2D([], [], marker="o", ls="none", ms=3.6,
                                   mfc=STAGE_COLOR[s], mec="none",
                                   label=STAGE_LABEL[s]) for s in STAGES],
                   fontsize=5.4, loc="lower left", handletextpad=0.3)
    fig.text(0.5, 0.015, "Integrated half only. The delivered raw-PCA UMAP is "
             "not staged and is not re-derivable without making a new "
             "embedding.", ha="center", va="bottom", fontsize=5.4)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    stem = os.path.join(out_dir, "SFig2E_umap_harmony")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_e(cells, out_dir):
    """QC covariates on the Harmony UMAP: total UMIs and detected genes."""
    d = _shuffled(cells)
    specs = [(np.log10(d["n_counts"].to_numpy()), "log10 total UMIs"),
             (np.log10(d["n_genes"].to_numpy()), "log10 detected genes")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0))
    for ax, (v, lab) in zip(axes, specs):
        lo, hi = np.percentile(v, [1, 99])
        sc = ax.scatter(d["umap_x"], d["umap_y"], s=0.06, c=v, cmap="viridis",
                        vmin=lo, vmax=hi, linewidths=0, rasterized=True)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(lab, fontsize=7.5)
        cb = fig.colorbar(sc, ax=ax, fraction=0.038, pad=0.02)
        cb.ax.tick_params(labelsize=5.2)
        cb.outline.set_visible(False)
    fig.suptitle("QC covariates on the Harmony UMAP", fontsize=8.5)
    fig.text(0.5, 0.015, f"Two of the delivered four overlays: {QC_ABSENT}. "
             f"Colour is clipped to the 1st-99th percentile.", ha="center",
             va="bottom", fontsize=5.4)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94))
    stem = os.path.join(out_dir, "extra_qc_umap_overlays")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def _compartment_totals(wide):
    """Per-replicate counts summed into the four compartments."""
    out = pd.DataFrame(index=wide["replicate"].values)
    for c in COMPARTMENTS:
        cols = [t for t in TYPE_ORDER if TYPE_COMPARTMENT[t] == c]
        out[c] = wide[cols].to_numpy().sum(axis=1)
    return out


def panel_f(counts, out_dir):
    """Cells recovered per replicate, stacked by compartment, log axis."""
    d = counts.set_index("replicate").reindex(
        sorted(counts["replicate"], key=lambda r: (STAGES.index(
            counts.set_index("replicate").loc[r, "stage"]), r))).reset_index()
    comp = _compartment_totals(d)
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    bottom = np.zeros(len(d))
    for c in COMPARTMENTS:
        ax.bar(range(len(d)), comp[c].values, bottom=bottom, width=0.7,
               color=COMPARTMENT_COLOR[c], edgecolor="black", lw=0.4, label=c)
        bottom += comp[c].values
    for i, v in enumerate(bottom):
        ax.text(i, v * 1.02, f"{int(v):,}", ha="center", va="bottom",
                fontsize=5.2, rotation=90)
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([r.replace("_", "-") for r in d["replicate"]], rotation=90,
                       fontsize=5.8)
    ax.set_ylabel("Cells recovered")
    ax.set_ylim(0, bottom.max() * 1.35)
    ax.set_title("Cells recovered per replicate", fontsize=8.5)
    ax.legend(fontsize=5.6, ncol=4, loc="upper left")
    ax.text(0.99, 0.95, f"{int(bottom.sum()):,} cells over {len(d)} replicates",
            transform=ax.transAxes, ha="right", va="top", fontsize=5.8)
    fig.tight_layout()
    stem = os.path.join(out_dir, "SFig2G_cells_per_replicate")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_g(pct, palette, out_dir):
    """Within-replicate composition, the delivered percentages as given."""
    d = pct.set_index("replicate")
    order = [r for s in STAGES for r in d.index[d["stage"] == s]]
    d = d.reindex(order)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    bottom = np.zeros(len(d))
    for t in TYPE_ORDER:
        v = d[t].to_numpy(dtype=float)
        ax.bar(range(len(d)), v, bottom=bottom, width=0.72, color=palette[t],
               edgecolor="white", lw=0.25, label=TYPE_SHORT.get(t, t))
        bottom += v
    ax.set_xticks(range(len(d)))
    ax.set_xticklabels([r.replace("_", "-") for r in d.index], rotation=90,
                       fontsize=5.8)
    ax.set_ylabel("% of the replicate")
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.7, len(d) - 0.3)
    ax.set_title("Within-replicate composition", fontsize=8.5)
    ax.legend(fontsize=4.8, loc="center left", bbox_to_anchor=(1.005, 0.5),
              handletextpad=0.35, labelspacing=0.22)
    fig.tight_layout()
    stem = os.path.join(out_dir, "SFig2H_composition_per_replicate")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


def panel_h(equal_n, boot, stats, out_dir):
    """Equal-N subsampling against full depth, and the bootstrap share spread."""
    d = equal_n.sort_values("N_per_replicate", ascending=False).reset_index(drop=True)
    types = list(d["cell_type"])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4),
                             gridspec_kw={"width_ratios": [1.15, 1.0]})
    ax = axes[0]
    y = np.arange(len(d))
    ax.hlines(y, d["equalN_p05"], d["equalN_p95"], lw=2.2, color="#bbbbbb",
              zorder=1)
    ax.plot(d["full_depth"], y, ls="none", marker="D", ms=3.6, mfc="white",
            mec="black", mew=0.7, label="full depth", zorder=3)
    ax.plot(d["equalN_median"], y, ls="none", marker="o", ms=4.4,
            mfc="#367f80", mec="black", mew=0.5, label="equal-N median", zorder=4)
    # get_yaxis_transform puts x in axes fraction and y in data coordinates, so
    # the label sits just outside the axes at its own row.
    for i, r in d.iterrows():
        ax.text(1.01, i, f"{r['retained_%']:.0f}%",
                transform=ax.get_yaxis_transform(), va="center", fontsize=5.2)
    ax.set_yticks(y)
    ax.set_yticklabels([TYPE_SHORT.get(t, t) for t in types], fontsize=5.8)
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Cells")
    ax.set_title("Equal-N against full depth (% retained at right)", fontsize=7.5)
    ax.legend(fontsize=5.4, loc="lower right")

    ax = axes[1]
    share_cols = [c for c in boot.columns if c not in ("panel", "draw")]
    for i, t in enumerate(types):
        if t not in share_cols:
            continue
        v = boot[t].to_numpy(dtype=float)
        ax.plot(v, np.full(v.size, i), ls="none", marker="o", ms=2.2,
                mfc="#4682B4", mec="none", alpha=0.65)
    ax.set_yticks(np.arange(len(types)))
    ax.set_yticklabels([])
    ax.invert_yaxis()
    ax.set_xlabel("% of the equal-N draw")
    ax.set_title(f"All {len(boot)} bootstrap draws, rows as at left", fontsize=7.5)
    rho = stats.loc["spearman_rho_shares_fulldepth_vs_equalN", "value"]
    note = stats.loc["spearman_rho_shares_fulldepth_vs_equalN", "note"]
    ax.text(0.98, 0.03, f"shares full depth against equal-N: Spearman "
            f"rho {float(rho):.2f} ({note})", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=5.4)
    fig.suptitle("Equal-N subsampling: depth-limited, not composition-neutral",
                 fontsize=8.5)
    # The note already reads 'of 16; missing: ...', so only the count is prefixed.
    covered = int(float(stats.loc["equalN_types_covered", "value"]))
    fig.text(0.02, 0.015, f"{covered} types covered "
             f"{stats.loc['equalN_types_covered', 'note']}", fontsize=5.2,
             ha="left", va="bottom")
    fig.tight_layout(rect=(0, 0.045, 1, 0.94))
    stem = os.path.join(out_dir, "extra_equalN_subsampling")
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
    fig, ax = plt.subplots(figsize=(2.6, 1.9))
    ax.axis("off")
    ax.add_patch(Rectangle((0.01, 0.01), 0.98, 0.98, transform=ax.transAxes,
                           facecolor="none", edgecolor="#999999", lw=0.8,
                           ls=(0, (4, 3))))
    ax.text(0.5, 0.87, f"Panel {label}", transform=ax.transAxes, ha="center",
            va="center", fontsize=9, fontweight="bold", color="#555555")
    ax.text(0.5, 0.45, "\n".join(_wrap(text, 38)), transform=ax.transAxes,
            ha="center", va="center", fontsize=5.2, color="#555555")
    ax.text(0.5, 0.05, "not computable from any table on disk",
            transform=ax.transAxes, ha="center", va="center", fontsize=5.4,
            style="italic", color="#888888")
    stem = os.path.join(out_dir, f"SFig2{label}_NOTCOMPUTABLE_placeholder")
    render_svg.save_svg(fig, stem, also_pdf=False)
    plt.close(fig)
    return stem + ".svg"


def build(repo_root: str, out_dir: str) -> dict:
    """Render the computable SFig2 panels; slug the MCA and raw-PCA ones."""
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    mpl.rcParams["savefig.dpi"] = 300  # resolution of the rasterized scatter layers
    tab = load_inputs(repo_root)

    paths = {}
    paths["a"] = panel_a(tab["mz"], out_dir)
    for lab, reason in NOT_COMPUTABLE.items():
        _slug(lab, reason, out_dir)
        paths[lab] = f"NOT COMPUTABLE: {reason}"
    paths["d"] = panel_d(tab["cells"], tab["palette"], out_dir)
    paths["e"] = panel_e(tab["cells"], out_dir)
    paths["f"] = panel_f(tab["counts"], out_dir)
    paths["g"] = panel_g(tab["pct"], tab["palette"], out_dir)
    paths["h"] = panel_h(tab["equal_n"], tab["boot"], tab["stats"], out_dir)
    return paths
