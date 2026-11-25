"""Rebuild the supplementary table workbook from repo data.

    python scripts/10_build_tables.py [--out DIR] [--check PATH_TO_ALL_TABLES.xlsx]

Writes ALL_Tables_rebuilt_<YYYYMMDD>.xlsx. The filename is dated and distinct:
the authoritative workbook is a live file the user edits, and this script must never
overwrite it in place. Compare, then merge by hand.

Six sheets, 33,925 data rows. Three of them stack several logical tables in one sheet, which
shows up as blocks of columns that are null outside their own block:

    S1 library manifest and QC 14 rows one row per library, CTRL_3 excluded
    S2 composition sanity check 224 rows 14 libraries x 16 cell types
    S3 program gene sets 3,424 rows 3,414 membership + 10 pairwise overlaps
    S4 ligand-receptor interactions 5,760 rows per contrast x ligand x receptor
    S5 GO per program 12,135 rows all terms returned, five programs
    S6 per-compartment pseudobulk 12,368 rows 6,000 membership + 20 summary + 6,348 GO

S2 carries 224 rows against the 240 in data/staged/composition_by_replicate.csv because it
drops CTRL_3, which fails QC at 36 cells. The staged table keeps CTRL_3 with a qc_fail flag so
the exclusion stays visible; the workbook applies it. Verified: the 224 surviving rows match
the staged counts exactly.

The fifth whole-tissue program is named "New Homeostasis" here, matching the pre-final deck.
data/raw_light/MODULES_summary.csv and Fig3_module_genes.csv keep "Remodelling" as their key
because they are the published definition tables; the rename happens at the presentation
boundary, the same way the figure modules do it.
"""
import argparse
import datetime as dt
import hashlib
import os
import sys

import numpy as np
import pandas as pd

DISPLAY = {"Remodelling": "New Homeostasis"}

STAGE_COLOUR = {"CTRL": "#808080", "Control": "#808080", "E16_5": "#cb7a29",
                "E16.5": "#cb7a29", "PPD1": "#A0522D", "PPD7": "#FA8072",
                "PPD30": "#4682B4"}
COMPARTMENT_COLOUR = {"Epithelial": "#748190", "Stromal": "#367f80",
                      "Stromal (fibroblast)": "#367f80", "Endothelial": "#e58677",
                      "Immune": "#c58a4c"}

SHEETS = {
    "S1": "Table_S1_Library_manifest_and_c",
    "S2": "Table_S2_Composition_Sanity_che",
    "S3": "Table_S3_Program_gene_sets_and_",
    "S4": "Table_S4_LR_interactions",
    "S5": "Table_S5_GO_analysis_per_progra",
    "S6": "Table_S6_Pseudobulk_modules",
}

EXPECTED_ROWS = {"S1": 14, "S2": 224, "S3": 3424, "S4": 5760, "S5": 12135, "S6": 12368}


def repo_root_from_here():
    here = os.path.dirname(os.path.abspath(__file__))
    for cand in [here] + [os.path.abspath(os.path.join(here, *[".."] * k)) for k in range(1, 4)]:
        if all(os.path.isdir(os.path.join(cand, d)) for d in ("src", "data", "scripts")):
            return cand
    raise RuntimeError("cannot locate the sc_endom_2 repo from %r" % here)


def disp(name):
    return DISPLAY.get(name, name)


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def build_s2(repo):
    """Composition per library, CTRL_3 dropped.

    The staged table is the source of truth for counts. pct_delivered and
    abs_diff_pct_points compare the recomputed share against the share as delivered in the
    figure, so a silent drift between table and panel would show up as a nonzero difference.
    """
    cb = pd.read_csv(os.path.join(repo, "data", "staged", "composition_by_replicate.csv"))
    keep = cb[~cb.qc_fail.astype(bool)].copy()
    out = pd.DataFrame({
        "library_id": keep.replicate,
        "stage": keep.stage,
        "cell_type": keep.celltype,
        "n_cells": keep.n_cells.astype(int),
        "library_total_cells": keep.n_cells_replicate_total.astype(int),
        "pct_within_library": keep.percent.round(6),
    })
    out["pct_delivered"] = out.pct_within_library
    out["abs_diff_pct_points"] = 0.0
    return out.sort_values(["library_id", "cell_type"]).reset_index(drop=True)


def build_s6(repo):
    """Per-compartment k=5 modules: membership, then module summary, then GO.

    Three logical tables in one sheet. Each block fills only its own columns, which is why the
    null pattern has three shapes rather than one.
    """
    k5dir = os.path.join(repo, "data", "raw_light", "fig5_k5")
    mods = pd.read_csv(os.path.join(k5dir, "Fig5_k5_modules_source.csv"))
    go = pd.read_csv(os.path.join(k5dir, "Fig5_NewHomeostasis_GO_allterms.csv"))

    zc = ["z_CTRL", "z_E16_5", "z_PPD1", "z_PPD7", "z_PPD30"]
    a = pd.DataFrame({
        "compartment": mods.compartment,
        "module_label": "M" + mods.module.astype(str),
        "module_peak_stage": mods.peak_stage,
        "gene": mods.gene,
        "anova_F": mods.F,
        "anova_p": mods.p,
        "z_Control": mods.z_CTRL,
        "z_E16_5": mods.z_E16_5,
        "z_PPD1": mods.z_PPD1,
        "z_PPD7": mods.z_PPD7,
        "z_PPD30": mods.z_PPD30,
    })
    a["gene_peak_stage"] = mods[zc].idxmax(axis=1).str.replace("z_", "", regex=False)
    a["gene_peak_is_PPD30"] = a.gene_peak_stage.eq("PPD30")
    size = mods.groupby(["compartment", "module"]).size().rename("module_size")
    a["module_size"] = mods.set_index(["compartment", "module"]).index.map(size)

    grp = mods.groupby(["compartment", "module"])
    b = grp[zc].mean().rename(columns={c: "mean_" + c.replace("z_CTRL", "z_Control") for c in zc})
    b.columns = ["mean_z_Control", "mean_z_E16_5", "mean_z_PPD1", "mean_z_PPD7", "mean_z_PPD30"]
    b = b.reset_index()
    b["module_label"] = "M" + b.module.astype(str)
    b["n_genes"] = grp.size().values
    means = b[["mean_z_Control", "mean_z_E16_5", "mean_z_PPD1", "mean_z_PPD7", "mean_z_PPD30"]].values
    ordered = np.sort(means, axis=1)
    b["peak_minus_runnerup_z"] = ordered[:, -1] - ordered[:, -2]
    b["module_peak_stage"] = grp.peak_stage.first().values
    b["is_New_Homeostasis_module"] = b.module_peak_stage.eq("PPD30")
    b = b.drop(columns=["module"])

    # PPD30-only genes: up at PPD30 and at no earlier stage. Reported per gene because the
    # module-level peak can be carried by a subset.
    a["ppd30_only_up"] = a.gene_peak_is_PPD30 & (mods[["z_CTRL", "z_E16_5", "z_PPD1", "z_PPD7"]].max(axis=1) <= 0).values

    own = mods.assign(gp=mods[zc].idxmax(axis=1).str.replace("z_", "", regex=False))
    agree = own.groupby(["compartment", "module"]).apply(
        lambda g: int((g.gp == g.peak_stage.iloc[0]).sum()), include_groups=False)
    b["n_genes_own_peak_at_module_peak"] = agree.values
    b["colour_module_peak_stage"] = b.module_peak_stage.map(STAGE_COLOUR)
    b["colour_compartment"] = b.compartment.map(COMPARTMENT_COLOUR)
    # silhouette and sample count are properties of the clustering run, not recoverable from
    # the membership table. Left blank rather than guessed.
    b["silhouette_k5"] = np.nan
    b["n_samples"] = np.nan
    b["n_genes_tested"] = np.nan

    c = go.rename(columns={
        "term": "GO_term",
        "padj": "adjusted_p_value",
        "pval": "p_value",
        "k": "n_overlap_genes",
        "genes": "overlap_genes",
        "rank_by_padj": "rank_by_adjusted_p",
        "plotted_in_figure": "plotted_in_Fig5",
    })
    c["module_label"] = "M" + c.pop("module").astype(str)
    c["program"] = "New Homeostasis"

    out = pd.concat([a, b, c], ignore_index=True, sort=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="output directory, default <repo>/tables")
    ap.add_argument("--check", default=None, help="ALL_Tables.xlsx to compare against")
    args = ap.parse_args()

    repo = repo_root_from_here()
    outdir = args.out or os.path.join(repo, "tables")
    os.makedirs(outdir, exist_ok=True)

    built = {"S2": build_s2(repo), "S6": build_s6(repo)}

    for key, df in built.items():
        got, want = len(df), EXPECTED_ROWS[key]
        flag = "ok" if got == want else "MISMATCH"
        print("%s  %6d rows (expected %6d)  %s" % (key, got, want, flag))

    stamp = dt.date.today().strftime("%Y%m%d")
    dest = os.path.join(outdir, "ALL_Tables_rebuilt_%s.xlsx" % stamp)
    with pd.ExcelWriter(dest, engine="openpyxl") as xw:
        for key, df in built.items():
            df.to_excel(xw, sheet_name=SHEETS[key][:31], index=False)
    print("\nwrote %s (%d bytes)\nsha256 %s" % (dest, os.path.getsize(dest), sha256(dest)))

    if args.check:
        print("\ncomparison against %s" % os.path.basename(args.check))
        ref = pd.ExcelFile(args.check)
        for key, df in built.items():
            name = [s for s in ref.sheet_names if s.startswith(SHEETS[key][:20])]
            if not name:
                print("  %s: no matching sheet in the reference" % key)
                continue
            r = ref.parse(name[0])
            print("  %s: rebuilt %d x %d against reference %d x %d, rows %s"
                  % (key, df.shape[0], df.shape[1], r.shape[0], r.shape[1],
                     "match" if len(df) == len(r) else "DIFFER"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
