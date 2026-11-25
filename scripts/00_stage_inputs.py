#!/usr/bin/env python
"""
00_stage_inputs.py: derive every atlas-dependent figure input from the h5ad, once.

Every number that reaches Fig. 2, 3 and 4 and is not already a shipped light table
is produced here, from endometrium_annotated_v1.h5ad and nothing else.

Why a separate staging step
---------------------------
The atlas is 14.3 GB on disk: 698,631 cells x 17,987 genes stored as CSR int64 counts
with 1.14e9 stored nonzeros (X/data alone is about 9.1 GB, X/indices 4.6 GB). On an
8 GB machine anndata.read_h5ad() and any densification are out of reach, so everything
below goes through h5py and reads X in row blocks, using X/indptr to find each block's
slice of data/indices. Peak memory is one block's nonzeros. The notebooks never open the
h5ad: they load the small tables written here, so the figures can be re-rendered and the
numbers checked without the 14.3 GB file.

Outputs (into --out, default data/staged)
----------------------------------------
  cells.parquet                 one row per cell: umap, stage, replicate, celltype,
                                compartment, n_counts, n_genes        -> Fig. 2 A, B, C, E
  composition_by_replicate.csv  cells per (replicate, celltype) and within-replicate
                                percent                               -> Fig. 2 F
  percell_module_scores.npz     per-cell mean log1p-CPM score for the five programs,
                                plus the coordinates and keys needed to plot them
                                                                      -> Fig. 3 A, extras
  marker_zscore.csv             16 cell types x marker genes, gene-wise z of mean
                                log1p-CPM                             -> Fig. 2 D

Usage
-----
  python scripts/00_stage_inputs.py --atlas /path/to/endometrium_annotated_v1.h5ad \\
                                    --out data/staged
"""

from __future__ import annotations

import argparse
import json
import os

import h5py
import numpy as np
import pandas as pd

# Definition layer. These constants are PUBLISHED choices, not tuning knobs, the module
# definitions, their plotting order and the compartment collapse were fixed upstream and
# the figures' claims are stated in their terms. They are written out here so a reader can
# see them without opening the notebook, but they must not be "improved" to make a number
# agree: if a count disagrees, the mapping is right and the assumption is wrong.


# Order is load-bearing: row order of percell_module_scores.npz['scores'] and the order the
# five wave panels appear in Fig3b-f.
MODULE_ORDER = ["Homeostasis", "Pregnancy", "Early repair", "Late repair", "Remodelling"]

# The fifth module is called "New Homeostasis" in the pre-final deck. That is a display
# label only: the gene set is unchanged (789 genes, peaks PPD30), and the published
# definition tables Fig3_module_genes.csv and MODULES_summary.csv are keyed on
# "Remodelling". Staging reads those tables, so MODULE_ORDER above keeps the published key
# and names[] in percell_module_scores.npz is written with it; src/fig3_panels.py and
# src/fig4_panels.py map to the display name where they draw. Renaming the key here would
# break the module-gene lookup below and the npz name check in fig3_panels.panel_g.
MODULE_DISPLAY = {"Remodelling": "New Homeostasis"}

# Order is load-bearing: stage_codes in the npz index into this, and every stage axis in
# Fig2/Fig3/Fig4 runs Control -> pregnancy -> postpartum day 1/7/30 in real time order,
# which is not the alphabetical order h5py hands back.
STAGE_ORDER = ["CTRL", "E16_5", "PPD1", "PPD7", "PPD30"]

# Library-size normalisation target for the log1p transform used by every module score and
# by the Fig2d marker matrix. This is part of the published definition (scanpy
# normalize_total default), see stream_expression() for why it is not a free rescaling.
TARGET_SUM = 1e4

# The 4-way compartment collapse of the 16 named cell types. Stromal is the
# residual class (it absorbs the fibroblast/pericyte/mesothelial/smooth-muscle families), so
# it is computed as "everything not named here", that way a cell type appearing in the map
# but forgotten here can never be dropped, it just lands in Stromal, and the
# assertion below (16 types, 4 compartments, no NaN) still has to hold.
COMPARTMENT_EXPLICIT = {
    "Epithelial": ["Epithelial"],
    "Endothelial": ["Vascular endothelial", "Lymphatic endothelial"],
    "Immune": [
        "Macrophage",
        "Dendritic cell",
        "Macrophage/APC",
        "Dendritic cell (cDC1)",
        "T cell",
        "NK cell",
    ],
}
COMPARTMENT_RESIDUAL = "Stromal"

# QC: this replicate yielded 36 cells and cannot support a replicate-level mean, so it is
# flagged (not deleted) here and excluded downstream wherever a per-replicate statistic is
# formed. It is kept in cells.parquet because the per-cell UMAP panels do show its cells.
QC_FAIL_REPLICATES = ["CTRL_3"]

# Expected sizes, asserted, never inferred. A mismatch means the wrong h5ad was passed.
N_CELLS = 698_631
N_GENES = 17_987
STAGE_COUNTS = {
    "CTRL": 127_161,
    "E16_5": 17_688,
    "PPD1": 194_125,
    "PPD7": 338_578,
    "PPD30": 21_079,
}

# Cells per row block when streaming X. WHY this number: mean 1,635 stored nonzeros per
# cell, so 25k cells is ~41M nonzeros = ~330 MB for int64 data + ~164 MB for int32 indices.
# That leaves the 8 GB ceiling comfortable while keeping the number of h5py reads (28) low;
# the read is sequential on disk, which is what dominates the runtime.
BLOCK = 25_000


def read_str_array(h5obj) -> np.ndarray:
    """h5py hands back bytes for variable-length strings; figures need str."""
    a = h5obj[:]
    return np.array([x.decode() if isinstance(x, bytes) else x for x in a])


def read_categorical(g: h5py.Group) -> tuple[np.ndarray, np.ndarray]:
    """AnnData categorical -> (categories, codes). Kept explicit because obs mixes plain
    int64 columns with categorical ones and confusing the two is the classic error here:
    RNA_snn_res.0.2 is a PLAIN int64 cluster id, while timepoint/replicate are categorical."""
    return read_str_array(g["categories"]), g["codes"][:]


def load_celltype_map(raw_light: str) -> pd.DataFrame:
    """Cluster -> named cell type, from the shipped group map.

    WHY not obs['cluster_label']: that column holds stale provisional strings
    ('7 (Epithelial)', '20 (Ambiguous(SmoothMuscle?))') from an earlier annotation round.
    The 16 names used in the paper exist only in paper2/group_map_17.csv, keyed on the
    res-0.2 clustering. Cluster 20 in particular resolves to Smooth muscle here, where the
    stale label guesses at an immune identity, so deriving names from obs would change Fig2c/d/f. We therefore map through the CSV and require full coverage.
    """
    gm = pd.read_csv(os.path.join(raw_light, "group_map_17.csv"))
    need = {"cluster", "label", "marker_signature"}
    missing = need - set(gm.columns)
    if missing:
        raise SystemExit(f"group_map_17.csv missing columns: {sorted(missing)}")
    gm["cluster"] = gm["cluster"].astype(int)
    if gm["cluster"].duplicated().any():
        raise SystemExit("group_map_17.csv has duplicate cluster rows")
    return gm


def compartment_of(label: str) -> str:
    for comp, members in COMPARTMENT_EXPLICIT.items():
        if label in members:
            return comp
    return COMPARTMENT_RESIDUAL


def build_cells(f: h5py.File, gm: pd.DataFrame) -> pd.DataFrame:
    """cells.parquet, the per-cell table every UMAP panel and the composition bars read.

    Everything here comes from obs/obsm, which are small (a few hundred MB at most), so this
    part needs no streaming. n_counts/n_genes are taken from the stored nCount_RNA /
    nFeature_RNA rather than recomputed, so that the table reports the same QC numbers the
    upstream Seurat object was filtered on.
    """
    umap = f["obsm/X_umap"][:]
    if umap.shape != (N_CELLS, 2):
        raise SystemExit(f"unexpected X_umap shape {umap.shape}")

    stage_cats, stage_codes = read_categorical(f["obs/timepoint"])
    rep_cats, rep_codes = read_categorical(f["obs/replicate"])
    clusters = f["obs/RNA_snn_res.0.2"][:].astype(int)

    unmapped = sorted(set(np.unique(clusters)) - set(gm["cluster"]))
    if unmapped:
        # Fatal: an unmapped cluster would appear as NaN cell type and
        # quietly shrink Fig2c/f rather than announcing itself.
        raise SystemExit(f"clusters absent from group_map_17.csv: {unmapped}")

    lab = dict(zip(gm["cluster"], gm["label"]))
    celltype = np.array([lab[c] for c in clusters])
    compartment = np.array([compartment_of(x) for x in celltype])

    cells = pd.DataFrame(
        {
            "umap_x": umap[:, 0].astype("float32"),
            "umap_y": umap[:, 1].astype("float32"),
            "stage": pd.Categorical(stage_cats[stage_codes], categories=STAGE_ORDER),
            "replicate": pd.Categorical(rep_cats[rep_codes], categories=list(rep_cats)),
            "cluster": clusters.astype("int16"),
            "celltype": celltype,
            "compartment": pd.Categorical(
                compartment,
                categories=list(COMPARTMENT_EXPLICIT) + [COMPARTMENT_RESIDUAL],
            ),
            "n_counts": f["obs/nCount_RNA"][:].astype("int32"),
            "n_genes": f["obs/nFeature_RNA"][:].astype("int32"),
        }
    )
    return cells


def build_composition(cells: pd.DataFrame) -> pd.DataFrame:
    """composition_by_replicate.csv, feeds Fig2f.

    Percent is WITHIN replicate (each replicate's cell types sum to 100), because Fig2f
    asks "what fraction of this animal's uterus is this cell type" and the replicates differ
    in total cells by more than an order of magnitude. Missing (replicate, celltype) pairs
    are filled with 0 rather than dropped: an absent cell type is a real observation of
    zero, and leaving the row out would bias the across-replicate mean upward.
    """
    reps = list(cells["replicate"].cat.categories)
    types = sorted(cells["celltype"].unique())
    idx = pd.MultiIndex.from_product([reps, types], names=["replicate", "celltype"])
    n = (
        cells.groupby(["replicate", "celltype"], observed=True)
        .size()
        .reindex(idx, fill_value=0)
        .rename("n_cells")
        .reset_index()
    )
    tot = n.groupby("replicate", observed=True)["n_cells"].transform("sum")
    n["n_cells_replicate_total"] = tot
    n["percent"] = np.where(tot > 0, n["n_cells"] / tot * 100.0, np.nan)
    n["stage"] = pd.Categorical(
        ["_".join(r.split("_")[:-1]) for r in n["replicate"]], categories=STAGE_ORDER
    )
    n["qc_fail"] = n["replicate"].isin(QC_FAIL_REPLICATES)
    return n[
        [
            "stage",
            "replicate",
            "celltype",
            "n_cells",
            "n_cells_replicate_total",
            "percent",
            "qc_fail",
        ]
    ]


def marker_gene_order(raw_light: str, gm: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Row (cell type) and column (gene) order for the Fig2d marker heatmap.

    WHY read an order from a file instead of sorting: the panel's readability rests entirely
    on its diagonal block structure, each cell type's own markers sit under its row, and
    that ordering was chosen by hand in the assembled deck. Sorting alphabetically, or
    taking group_map order, destroys the diagonal and makes the panel unreadable even though
    every number in it is identical.

    Fig2d_deck_marker_order.json supplies that order. Its "genes" array is read
    POSITIONALLY from the rendered panel
    (2026_RampUp/sc_endom_EDITABLE_SVG/Fig1/03_celltype_marker_heatmap.svg): the
    x-coordinate of each rotate(-90) <text> element, sorted ascending. PDF text
    extraction returns reading order, not visual column order, and must not be used.
    Rows in that same rendered panel are ALPHABETICAL top-to-bottom (read from each
    ytick label's y-coordinate), and alphabetical rows are what reproduce its
    diagonal-block score of |rho| = 0.771.

    The GENE SET here is still group_map_17.csv 'marker_signature' (66 unique genes);
    the JSON only supplies order, and it lists a wider 139-gene set. Signature genes
    absent from it are placed immediately after the last already-placed gene of the
    same cell type, so each joins its own diagonal block instead of landing at the end.

    NOTE: this 66-gene matrix (marker_zscore.csv) is not what the deck's Fig2d draws.
    The shipped panel uses all 139 genes via scripts/stage_marker_deck139.py ->
    data/staged/marker_zscore_deck139.csv. This function is kept so the original
    staging output stays reproducible.
    """
    with open(os.path.join(raw_light, "Fig2d_deck_marker_order.json")) as fh:
        obj = json.load(fh)
    deck = {"cols": obj["genes"] if isinstance(obj, dict) else obj, "rows": []}

    # Per-type signature genes, in the order the group map writes them.
    per_type: dict[str, list[str]] = {}
    for label, sig in zip(gm["label"], gm["marker_signature"]):
        per_type.setdefault(label, [])
        for g in str(sig).split("/"):
            g = g.strip()
            if g and g not in per_type[label]:
                per_type[label].append(g)

    sig_genes: list[str] = []
    for label in per_type:
        for g in per_type[label]:
            if g not in sig_genes:
                sig_genes.append(g)

    cols = [g for g in deck["cols"] if g in sig_genes]
    for label, genes in per_type.items():
        for g in genes:
            if g in cols:
                continue
            placed = [c for c in genes if c in cols]
            if placed:
                cols.insert(cols.index(placed[-1]) + 1, g)
            else:
                cols.append(g)

    # Rows are alphabetical, matching the rendered reference panel.
    rows = sorted(gm["label"].unique())

    if sorted(cols) != sorted(sig_genes):
        raise SystemExit("marker column order does not cover the signature gene set")
    return rows, cols


def stream_expression(
    f: h5py.File,
    var: np.ndarray,
    module_genes: dict[str, list[str]],
    marker_cols: list[str],
    clusters: np.ndarray,
) -> dict[str, np.ndarray]:
    """Single streaming pass over X. Returns per-cell module scores, per-cell library size,
    and per-(cluster, marker gene) expression sums.

    Normalisation: counts -> counts-per-10,000 per cell -> log1p. A module score is the
    MEAN log1p-CP10K over that module's genes, zeros included. That last point is what makes
    the score computable from a sparse stream at all: log1p(0) = 0, so a stored-nonzero-only
    sum divided by the module's full gene count equals the mean over all its genes. No
    scanpy score_genes control-gene binning is involved, this is the plain mean the
    module trajectories in Fig3/Fig4 were defined on.

    WHY 1e4 and not 1e6: TARGET_SUM is inside a log1p, so it is not a rescaling, it sets
    where the transform's compressive knee falls relative to typical counts, and 1e6 gives
    materially different per-cell scores (means ~3.2-3.6x larger, and only r ~= 0.95-0.97
    against the shipped scores). 1e4, scanpy's `normalize_total` default, which the
    upstream analysis used, reproduces the shipped per-cell scores exactly (r = 1.000000
    per module). Do not change it.

    Library size is recomputed from X rather than read from nCount_RNA, because the CPM
    denominator must be the total of the matrix actually being normalised; the two agree
    here, and that agreement is reported by the caller as a check rather than assumed.
    """
    n_cells = len(clusters)
    gene_ix = {g: i for i, g in enumerate(var)}

    # Compact index over the union of every gene we need, so the per-block membership test
    # is one array lookup instead of a Python set test per stored nonzero (1.14e9 of them).
    union: list[str] = []
    for m in MODULE_ORDER:
        for g in module_genes[m]:
            if g not in union:
                union.append(g)
    for g in marker_cols:
        if g not in union:
            union.append(g)

    lut = np.full(len(var), -1, dtype=np.int32)  # global gene index -> compact index
    for k, g in enumerate(union):
        lut[gene_ix[g]] = k
    K = len(union)
    compact_ix = {g: k for k, g in enumerate(union)}

    # K x 5 membership weights. Modules can share genes (they are defined by separate
    # contrasts), so this is a matrix rather than a single label per gene. Each column is
    # 1/n_genes_in_module, which turns the per-cell weighted sum straight into the mean.
    memb = np.zeros((K, len(MODULE_ORDER)), dtype=np.float32)
    for j, m in enumerate(MODULE_ORDER):
        genes = module_genes[m]
        for g in genes:
            memb[compact_ix[g], j] = 1.0 / len(genes)

    marker_slot = np.full(K, -1, dtype=np.int32)  # compact index -> marker column position
    for j, g in enumerate(marker_cols):
        marker_slot[compact_ix[g]] = j

    n_clust = int(clusters.max()) + 1
    n_mark = len(marker_cols)
    scores = np.zeros((len(MODULE_ORDER), n_cells), dtype=np.float32)
    libsize = np.zeros(n_cells, dtype=np.float32)
    marker_sums = np.zeros((n_clust, n_mark), dtype=np.float64)

    X = f["X"]
    indptr = X["indptr"][:]
    data_ds, ind_ds = X["data"], X["indices"]

    for start in range(0, n_cells, BLOCK):
        stop = min(start + BLOCK, n_cells)
        lo, hi = int(indptr[start]), int(indptr[stop])
        dat = data_ds[lo:hi].astype(np.float64)
        ind = ind_ds[lo:hi]
        # Row index of every stored nonzero in the block, from the indptr differences.
        rows = np.repeat(
            np.arange(stop - start, dtype=np.int64), np.diff(indptr[start : stop + 1])
        )

        # Library size uses all of the block's counts, including genes we do not keep
        # CPM must be relative to the whole transcriptome, not to the module subset.
        lib = np.bincount(rows, weights=dat, minlength=stop - start)
        libsize[start:stop] = lib

        keep = lut[ind] >= 0
        if keep.any():
            r = rows[keep]
            k = lut[ind[keep]]
            safe = np.where(lib > 0, lib, 1.0)
            v = np.log1p(dat[keep] / safe[r] * TARGET_SUM)

            for j in range(len(MODULE_ORDER)):
                w = memb[k, j]
                nz = w > 0
                if nz.any():
                    scores[j, start:stop] = np.bincount(
                        r[nz], weights=v[nz] * w[nz], minlength=stop - start
                    )

            ms = marker_slot[k]
            mk = ms >= 0
            if mk.any():
                flat = clusters[start:stop][r[mk]].astype(np.int64) * n_mark + ms[mk]
                marker_sums += np.bincount(
                    flat, weights=v[mk], minlength=n_clust * n_mark
                ).reshape(n_clust, n_mark)

        del dat, ind, rows

    return {"scores": scores, "libsize": libsize, "marker_sums": marker_sums}


def build_marker_zscore(
    marker_sums: np.ndarray,
    clusters: np.ndarray,
    gm: pd.DataFrame,
    rows: list[str],
    cols: list[str],
) -> pd.DataFrame:
    """marker_zscore.csv, Fig2d.

    Per-cluster sums are pooled to cell type first and divided by that type's TOTAL cell
    count, which is the mean over cells of the type (several named types span two clusters).
    The z-score is then taken gene-wise (down the column, across the 16 types) because the
    panel's question is "which cell type expresses this marker most", and genes differ in
    absolute expression by orders of magnitude; a row-wise z would instead rank markers
    within a type and wash out the diagonal.
    """
    lab = dict(zip(gm["cluster"], gm["label"]))
    n_by_cluster = np.bincount(clusters, minlength=marker_sums.shape[0])

    sums = pd.DataFrame(0.0, index=rows, columns=cols)
    ncell = pd.Series(0.0, index=rows)
    for c in range(marker_sums.shape[0]):
        if c not in lab:
            continue
        sums.loc[lab[c]] += marker_sums[c]
        ncell[lab[c]] += n_by_cluster[c]

    mean = sums.div(ncell, axis=0)
    z = (mean - mean.mean(axis=0)) / mean.std(axis=0, ddof=0).replace(0.0, np.nan)
    return z


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--atlas", required=True, help="endometrium_annotated_v1.h5ad")
    ap.add_argument("--out", default="data/staged")
    ap.add_argument(
        "--raw-light",
        default="data/raw_light",
        help="shipped light tables (group_map_17.csv, Fig3_module_genes.csv, ...)",
    )
    ap.add_argument(
        "--reference-npz",
        default=None,
        help="optional prior percell_module_scores.npz to validate against (Pearson r per module)",
    )
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    gm = load_celltype_map(args.raw_light)

    mg = pd.read_csv(os.path.join(args.raw_light, "Fig3_module_genes.csv"))
    module_genes = {m: mg.loc[mg["module"] == m, "gene"].tolist() for m in MODULE_ORDER}
    for m, g in module_genes.items():
        if not g:
            raise SystemExit(f"module {m} has no genes in Fig3_module_genes.csv")

    rows, cols = marker_gene_order(args.raw_light, gm)

    with h5py.File(args.atlas, "r") as f:
        shape = tuple(f["X"].attrs["shape"])
        if shape != (N_CELLS, N_GENES):
            raise SystemExit(f"unexpected atlas shape {shape}, expected {(N_CELLS, N_GENES)}")
        var = read_str_array(f["var/_index"])

        absent = [g for gl in module_genes.values() for g in gl if g not in set(var)]
        absent += [g for g in cols if g not in set(var)]
        if absent:
            raise SystemExit(f"{len(absent)} required genes absent from var: {absent[:10]}")

        cells = build_cells(f, gm)
        clusters = cells["cluster"].to_numpy().astype(np.int64)
        umap = cells[["umap_x", "umap_y"]].to_numpy(dtype=np.float32)
        cluster_cats, cluster_codes = read_categorical(f["obs/cluster_label"])

        agg = stream_expression(f, var, module_genes, cols, clusters)

    # verification. Fatal on disagreement.
    if len(cells) != N_CELLS:
        raise SystemExit(f"cells.parquet has {len(cells)} rows, expected {N_CELLS}")
    got = cells["stage"].value_counts().reindex(STAGE_ORDER).to_dict()
    bad = {k: (got[k], v) for k, v in STAGE_COUNTS.items() if got[k] != v}
    if bad:
        raise SystemExit(f"stage count mismatch (got, expected): {bad}")
    if cells["celltype"].nunique() != 16:
        raise SystemExit(f"{cells['celltype'].nunique()} distinct cell types, expected 16")
    if cells["compartment"].nunique() != 4:
        raise SystemExit(f"{cells['compartment'].nunique()} compartments, expected 4")
    if cells["celltype"].isna().any() or cells["compartment"].isna().any():
        raise SystemExit("NaN in celltype/compartment")

    cells.to_parquet(os.path.join(args.out, "cells.parquet"), index=False)

    comp = build_composition(cells)
    comp.to_csv(os.path.join(args.out, "composition_by_replicate.csv"), index=False)

    stage_codes = np.array(
        [STAGE_ORDER.index(s) for s in cells["stage"].astype(str)], dtype=np.int8
    )
    np.savez_compressed(
        os.path.join(args.out, "percell_module_scores.npz"),
        scores=agg["scores"].astype(np.float32),
        names=np.array(MODULE_ORDER),
        umap=umap,
        libsize=agg["libsize"].astype(np.float32),
        stage_cats=np.array(STAGE_ORDER),
        stage_codes=stage_codes,
        cluster_cats=cluster_cats,
        cluster_codes=cluster_codes.astype(np.int8),
    )

    z = build_marker_zscore(agg["marker_sums"], clusters, gm, rows, cols)
    z.to_csv(os.path.join(args.out, "marker_zscore.csv"))

    # report
    print("stage            staged     expected  ok")
    for s in STAGE_ORDER:
        print(f"{s:10s} {got[s]:10d} {STAGE_COUNTS[s]:12d}  {got[s] == STAGE_COUNTS[s]}")
    print(f"{'TOTAL':10s} {len(cells):10d} {N_CELLS:12d}  {len(cells) == N_CELLS}")
    print(f"\ncell types      : {cells['celltype'].nunique()} (expected 16)")
    print(f"compartments    : {sorted(cells['compartment'].unique())}")
    print(f"replicates      : {comp['replicate'].nunique()} "
          f"(qc_fail: {sorted(set(comp.loc[comp['qc_fail'], 'replicate']))})")
    lib_vs_counts = np.abs(agg["libsize"] - cells["n_counts"].to_numpy()).max()
    print(f"max |libsize - nCount_RNA| : {lib_vs_counts:g}  (0 = X totals match stored QC)")
    print("\nmodule score means:")
    for j, m in enumerate(MODULE_ORDER):
        # printed under the deck's display name, with the table key kept visible so the
        # reader can find the row in Fig3_module_genes.csv
        lab = MODULE_DISPLAY.get(m)
        lab = m if lab is None else f"{lab} [{m}]"
        print(f"  {lab:32s} {agg['scores'][j].mean():.4f}  "
              f"(n_genes={len(module_genes[m])})")
    print(f"\nmarker matrix   : {z.shape[0]} types x {z.shape[1]} genes")

    if args.reference_npz:
        with np.load(args.reference_npz, allow_pickle=True) as ref:
            print("\nPearson r vs reference npz (pass >= 0.98):")
            for j, m in enumerate(MODULE_ORDER):
                jr = list(ref["names"]).index(m)
                r = np.corrcoef(agg["scores"][j], ref["scores"][jr])[0, 1]
                print(f"  {m:14s} r = {r:.6f}  {'PASS' if r >= 0.98 else 'FAIL'}")

    print("\nwrote:")
    for fn in [
        "cells.parquet",
        "composition_by_replicate.csv",
        "percell_module_scores.npz",
        "marker_zscore.csv",
    ]:
        p = os.path.join(args.out, fn)
        print(f"  {fn:32s} {os.path.getsize(p) / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
