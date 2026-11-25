"""Programs resolved per compartment (Fig. 4A) and the program heatmap and GO dotplot (Fig. 3B, C).

    panels_ae(repo_root, out_dir) -> ({"a": svg_path}, concordance)   Fig. 4A
    panel_f(repo_root, out_dir)   -> (svg_path, info)                  Fig. 3B
    panel_g(repo_root, out_dir)   -> (svg_path, info)                  Fig. 3C

Fig. 4A  per-compartment program trajectories, one small panel per program with
         four compartment lines, via build_fig4_traj.py. Scoring each program inside
         each compartment separately rules out the trajectories being an artefact of
         which cell types are present at a stage.
Fig. 3B  program signature heatmap, via build_fig4_heatmap.py.
Fig. 3C  program GO dotplot (terms on y), via build_fig4_go.py.

This module loads the inputs, supplies the settings and owns the SVG export; the
three builders are called as shipped. Reads only from repo_root/data/.
"""
import os

import numpy as np
import pandas as pd

import render_svg
import build_fig4_traj
import build_fig4_heatmap
import build_fig4_go

STAGES_D = ["CTRL", "E16.5", "PPD1", "PPD7", "PPD30"]          # trajectory-table columns
STAGES_U = ["CTRL", "E16_5", "PPD1", "PPD7", "PPD30"]          # underscore vocabulary
STAGE_LAB = {"CTRL": "CTRL", "E16_5": "E16.5", "PPD1": "PPD1", "PPD7": "PPD7",
             "PPD30": "PPD30", "E16.5": "E16.5"}
MODULES = ["Homeostasis", "Pregnancy", "Early repair", "Late repair", "Remodelling"]
MOD_N = {"Homeostasis": 678, "Pregnancy": 217, "Early repair": 985,
         "Late repair": 745, "Remodelling": 789}
MOD_PEAK = {"Homeostasis": "CTRL", "Pregnancy": "E16_5", "Early repair": "PPD1",
            "Late repair": "PPD7", "Remodelling": "PPD30"}
MOD_COL = {"Homeostasis": "#808080", "Pregnancy": "#cb7a29", "Early repair": "#A0522D",
           "Late repair": "#FA8072", "Remodelling": "#4682B4"}
# Display labels for the pre-final deck. The fifth module is shown as "New Homeostasis";
# its gene set is unchanged (789 genes, peaks PPD30). The definition tables
# data/raw_light/MODULES_summary.csv and Fig3_module_genes.csv are published and keyed on
# "Remodelling", as are Fig4_module_gene_assignment.csv and Fig4_module_GO_plotted.csv,
# so "Remodelling" stays the internal key everywhere a table is looked up and the rename
# happens here, at the presentation boundary. Do not resolve the inconsistency by
# rewriting the CSV keys.
DISPLAY = {"Remodelling": "New Homeostasis"}


def disp(module):
    """The label a module is drawn with. Identity for the four unrenamed modules."""
    return DISPLAY.get(module, module)


COMPS = ["Epithelial", "Stromal", "Endothelial", "Immune"]
COMP_COL = {"Epithelial": "#748190", "Stromal": "#367f80",
            "Endothelial": "#e58677", "Immune": "#c58a4c"}
# The one documented exception to "every module peaks in its own stage in every
# compartment": Late repair in epithelium peaks at E16.5, not PPD7. That is the PPD7
# epithelial collapse, and it is a result, not a defect, panels a-e assert it below so a
# future rebuild that quietly loses it fails loudly.
KNOWN_PEAK_EXCEPTIONS = {("Late repair", "Epithelial"): "E16.5"}


def _raw_light(repo_root):
    return os.path.join(repo_root, "data", "raw_light")


# panels a-e: compartment trajectories
def panels_ae(repo_root, out_dir):
    """Five module panels x four compartment lines, via build_fig4_traj.build() verbatim.

    The builder wants TRAJ as {module: DataFrame indexed by compartment, columns=stages},
    with the gene count carried on .attrs['n'] because it prints it in the panel header.
    """
    rl = _raw_light(repo_root)
    tab = pd.read_csv(os.path.join(rl, "Fig4_module_trajectories_by_compartment.csv"))
    if len(tab) != 20:
        raise ValueError(f"expected 20 rows (5 modules x 4 compartments), got {len(tab)}")
    TRAJ, concordance = {}, []
    for m in MODULES:
        T = tab[tab.module == m].set_index("compartment").reindex(COMPS)[STAGES_D]
        if T.isna().any().any():
            raise ValueError(f"{m}: missing compartment rows {list(T[T.isna().any(1)].index)}")
        T.attrs["n"] = MOD_N[m]
        TRAJ[m] = T
        for c in COMPS:
            peak = STAGES_D[int(np.argmax(T.loc[c].values))]
            expected = STAGE_LAB[MOD_PEAK[m]]
            exc = KNOWN_PEAK_EXCEPTIONS.get((m, c))
            concordance.append(dict(module=m, compartment=c, peak=peak, expected=expected,
                                    concordant=peak == expected,
                                    documented_exception=exc is not None and peak == exc))
    conc = pd.DataFrame(concordance)
    bad = conc[~conc.concordant & ~conc.documented_exception]
    if len(bad):
        raise ValueError("peak concordance broke in a way that is NOT the documented "
                         f"epithelial Late-repair exception:\n{bad.to_string()}")
    missing_exc = [k for k, v in KNOWN_PEAK_EXCEPTIONS.items()
                   if not bool(conc[(conc.module == k[0]) & (conc.compartment == k[1])
                                    ].documented_exception.iloc[0])]
    if missing_exc:
        raise ValueError(f"documented exception(s) {missing_exc} no longer present in the "
                         f"data, the PPD7 epithelial collapse has changed, check the input")

    stem = os.path.join(out_dir, "Fig4A_module_trajectories_by_compartment")
    # The builder prints its `order` entries as the panel headers, so the display names go
    # in there and the three lookups it drives are re-keyed to match. Every table read
    # above used the internal keys; nothing downstream of this point indexes a CSV.
    TRAJ_D = {disp(m): TRAJ[m] for m in MODULES}
    fig = build_fig4_traj.build(
        TRAJ_D, [disp(m) for m in MODULES], {disp(m): MOD_COL[m] for m in MODULES},
        {disp(m): MOD_PEAK[m] for m in MODULES}, COMPS, COMP_COL, STAGES_U, STAGE_LAB,
        stem,
        "Each module runs its own course, and it runs it in every compartment",
        subtitle=("Every module peaks in its own stage in all four compartments \u2014 "
                  "except late repair in epithelium"),
        fs=2.0, lw=2.8, ms=8.5, save=render_svg.save_svg,
        pw_scale=1.18, rotation=45, legend_below=True)
    import matplotlib.pyplot as plt
    plt.close(fig)
    return {k: stem + ".svg" for k in "abcde"}, conc


# panel f: signature heatmap
def panel_f(repo_root, out_dir):
    """Gene-level module signature heatmap, via build_fig4_heatmap.build() verbatim.

    two shipped behaviours this wrapper is responsible for and must not lose:

    1. save trap, build() calls mpl.rcParams.update() internally, which resets
       svg.fonttype to 'path' and would bake every label into glyph outlines. It accepts
       a save= callback; passing render_svg.save_svg keeps export (and therefore live
       true-type text) owned by the helper. Verified by verify(): text_elements > 0.
    2. Column tracks above the matrix are only clusters (16 named types) and compartment
       (4). There is no state / dominant-programme track: the five row blocks
       ARE the state, and a third track would assert a per-column state call this panel
       does not make.
    """
    rl = _raw_light(repo_root)
    Erep, mods, cann, stc = build_fig4_heatmap.load_inputs(
        os.path.join(rl, "pseudobulk_rep_by_type.npz"),
        os.path.join(rl, "Fig4_module_gene_assignment.csv"),
        os.path.join(rl, "Fig3_heatmap_column_annotation.csv"),
        os.path.join(rl, "Fig3_column_state_calls.csv"))
    # the heatmap's module table is the PEAK-ASSIGNED subset (each gene assigned to the one
    # module whose stage it peaks in), so its per-module counts are smaller than the
    # canonical module sizes and the two must not be conflated
    n_by_mod = mods.module.value_counts().to_dict()
    # build_fig4_heatmap groups rows by PHASE and falls back to its own PHASE_MAP when the
    # table has no 'phase' column. That fallback is keyed to the retired k-means module ids
    # (M0-M5), so with the five NAMED modules it raises "no phase assignment". Supplying the
    # column is the builder's own documented extension point ("phase (optional)") and needs
    # no edit to it: in this vocabulary the module name IS the phase, one per peak stage.
    # Assert the mapping resolves rather than trusting it, so a future rename fails loudly.
    if "phase" not in mods.columns:
        mods = mods.assign(phase=mods.module)
    # Each row band prints BOTH its phase name and its module name, so the display rename
    # has to reach both columns of the caller-supplied table. The builder resolves a phase
    # to its peak stage through its own alias table, which is what sets the band's colour
    # and its position in course order; "New Homeostasis" is registered there as another
    # spelling of the PPD30 phase. build_fig4_heatmap.py itself stays byte-identical, and
    # the gene assignment on disk keeps its "Remodelling" key.
    build_fig4_heatmap.PHASE_ALIAS[disp("Remodelling")] = "PPD30"
    mods = mods.assign(module=mods.module.map(disp), phase=mods.phase.map(disp))
    unresolved = sorted({p for p in mods.phase.unique()
                         if build_fig4_heatmap.phase_stage(p) is None})
    if unresolved:
        raise ValueError(f"phases the builder cannot map to a stage: {unresolved}")
    stem = os.path.join(out_dir, "Fig3B_program_signature_heatmap")
    fig, blocks, col_order, gene_order = build_fig4_heatmap.build(
        Erep, mods, cann, stc, stem,
        "Each module of the postpartum course has its own gene signature",
        subtitle=(f"rows = {len(mods):,} module genes, each assigned to the module whose "
                  f"stage it peaks in"),
        fs=2.0, save=render_svg.save_svg)
    import matplotlib.pyplot as plt
    plt.close(fig)
    info = dict(n_rows=len(gene_order), n_cols=len(col_order),
                blocks=[(b["phase"], b["n"]) for b in blocks],
                n_by_module=n_by_mod)
    return stem + ".svg", info


# panel g: GO dotplot
def panel_g(repo_root, out_dir):
    """Module GO dotplot, portrait layout, via build_fig4_go.build() verbatim.

    Dot area = genes in term (size key 5/20/45), colour = -log10 adjusted P on viridis.
    The full x range stays visible: only 4 of the 30 terms exceed gene ratio 0.35, and
    clipping to the dense left-hand cluster would hide exactly the four terms where a
    module covers most of a pathway.
    """
    rl = _raw_light(repo_root)
    sel = pd.read_csv(os.path.join(rl, "Fig4_module_GO_plotted.csv"))
    # the table is already the selection (6 terms per module); re-ordering it here would
    # re-select, so only the within-module draw order is applied
    sel = pd.concat([sel[sel.wave == m].sort_values("gene_ratio", ascending=False)
                     for m in MODULES], ignore_index=True)
    per_mod = sel.wave.value_counts().to_dict()
    if set(per_mod) != set(MODULES) or set(per_mod.values()) != {6}:
        raise ValueError(f"expected 6 terms for each of the 5 modules, got {per_mod}")
    stem = os.path.join(out_dir, "Fig3C_program_GO_dotplot")
    # The builder colours each term by sel.wave and prints the band header from its `waves`
    # list, so the display rename is applied to the frame it draws from and to the two
    # dicts keyed alongside it. The selection above read the archived "Remodelling" key.
    sel = sel.assign(wave=sel.wave.map(disp))
    fig = build_fig4_go.build(
        sel, [disp(m) for m in MODULES], {disp(m): MOD_COL[m] for m in MODULES},
        {disp(m): MOD_PEAK[m] for m in MODULES}, STAGE_LAB, stem,
        "GO biological process, per module",
        group_word="module", fs=4.0, save=render_svg.save_svg,
        row_scale=2.4, x_scale=0.62)
    import matplotlib.pyplot as plt
    plt.close(fig)
    ax = fig.axes[0]
    info = dict(n_terms=len(sel), per_module=per_mod,
                gene_ratio_range=(round(float(sel.gene_ratio.min()), 4),
                                  round(float(sel.gene_ratio.max()), 4)),
                x_limits=tuple(round(float(v), 4) for v in ax.get_xlim()),
                n_terms_above_0p35=int((sel.gene_ratio > 0.35).sum()))
    return stem + ".svg", info


# driver
def build(repo_root: str, out_dir: str) -> dict:
    """Render every Fig4 panel into out_dir. Returns {panel_label: svg_path}."""
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    out, conc = panels_ae(repo_root, out_dir)
    out["f"], hinfo = panel_f(repo_root, out_dir)
    out["g"], ginfo = panel_g(repo_root, out_dir)
    build.report = dict(peak_concordance=conc, heatmap=hinfo, go=ginfo)
    return out


if __name__ == "__main__":
    import sys
    r = sys.argv[1]
    print(build(r, os.path.join(r, "figures")))
