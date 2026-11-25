"""Whole-tissue transcriptional programs (Fig. 3A and three extra panels).

    panels_bf(repo_root, out_dir) -> ({label: svg_path}, report)   Fig. 3A
    panel_a / panel_g / panel_h                                     extras

Fig. 3A  five program-score trajectories, one file per program in course order:
         coloured mean over about 350 sampled per-gene grey traces. Whole-tissue
         per-sample pseudobulk, log2(CPM + 1), z-scored per gene across the 14
         usable samples (CTRL_3 dropped).
extras   five-wave schematic (a drawing, not data); five per-stage UMAPs coloured by
         each cell's dominant program (within-cell z of the five scores, argmax,
         called at top z >= 0.5); the 16-type x 4-contrast DEG swarm matrix drawn by
         build_fig3n_swarm.py.

Reads only from repo_root/data/; the h5ad is never opened here.
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import render_svg
import build_fig3n_swarm as swarm_mod

# canonical layer
STAGES = ["CTRL", "E16_5", "PPD1", "PPD7", "PPD30"]
STAGE_LAB = {"CTRL": "CTRL", "E16_5": "E16.5", "PPD1": "PPD1", "PPD7": "PPD7", "PPD30": "PPD30"}
# The five canonical modules are a PUBLISHED definition layer: name, gene count, peak
# stage and colour are fixed. Nothing here recomputes them; the gene lists come from
# data/raw_light/Fig3_module_genes.csv and are checked against these counts.
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
# "Remodelling", so "Remodelling" stays the internal key everywhere a table is looked up
# and the rename happens here, at the presentation boundary. Do not resolve the
# inconsistency by rewriting the CSV keys.
DISPLAY = {"Remodelling": "New Homeostasis"}


def disp(module):
    """The label a module is drawn with. Identity for the four unrenamed modules."""
    return DISPLAY.get(module, module)


# Fig3g greys. Uncalled cells and other-stage cells must be visually distinct from each
# other AND from every module colour, or "no dominant module" reads as a sixth module.
GREY_UNCALLED = "#bfbfbf"
GREY_OTHER = "#ececec"
# Full per-stage cell counts (the atlas, not the display subsample). Printed on panel g.
STAGE_N = {"CTRL": 127161, "E16_5": 17688, "PPD1": 194125, "PPD7": 338578, "PPD30": 21079}
FG_CAP = 60000          # foreground display cap per stage panel (background stays complete)
N_TRACES = 350          # grey per-gene traces behind each trajectory
SEED = 11


def _paths(repo_root):
    return (os.path.join(repo_root, "data", "raw_light"),
            os.path.join(repo_root, "data", "staged"))


# panel a: schematic
def panel_a(out_dir):
    """Idealised programme-activity curves. Explicitly a schematic, no data is read."""
    fig, ax = plt.subplots(figsize=(4.6, 2.35))
    x = np.linspace(-0.55, 4.55, 700)
    for m in MODULES:
        mu = STAGES.index(MOD_PEAK[m])
        y = np.exp(-0.5 * ((x - mu) / 0.78) ** 2)
        ax.plot(x, y, color=MOD_COL[m], lw=2.0, zorder=3, solid_capstyle="round")
        # Label at the PEAK, not the right edge: three of the five curves have decayed to
        # ~0 by x=4.55, so right-edge labels stack on top of each other.
        ax.text(mu, 1.045, disp(m), color=MOD_COL[m], fontsize=7.0, va="bottom",
                ha="center", fontweight="bold")
    # parturition sits BETWEEN E16.5 and PPD1, it is an event, not a sampled stage.
    # Its label runs along the line so it cannot collide with the two nearest peak labels.
    ax.axvline(1.5, color="#444444", lw=1.0, ls=(0, (3.5, 2.2)), zorder=2)
    ax.text(1.47, 0.62, "parturition", fontsize=6.6, color="#444444", ha="right",
            va="center", rotation=90)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.04, 1.22)
    ax.set_xticks(range(5))
    ax.set_xticklabels([STAGE_LAB[s] for s in STAGES], fontsize=6.6, color="#555555")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["low", "high"], fontsize=6.6)
    ax.set_ylabel("relative programme activity", fontsize=7.0)
    ax.tick_params(length=2.0, pad=1.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.set_title("Five programmes, one course", fontsize=8.6, loc="left", pad=12)
    ax.text(0.0, 1.035, "schematic, idealised curves, not measured data",
            transform=ax.transAxes, fontsize=6.6, color="#777777", va="bottom")
    stem = os.path.join(out_dir, "extra_five_wave_schematic")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg"


# panels b-f: per-module trajectories
def _module_z(raw_light):
    """Per-gene z of log2(CPM+1) across the 14 usable whole-tissue samples.

    CTRL_3 is a 36-cell QC failure and is absent from the staged counts table; the
    assertion below keeps it that way rather than trusting the column list.
    """
    counts = pd.read_csv(os.path.join(raw_light, "Fig7_wholeTissue_persample_pseudobulk_counts.csv"),
                         index_col=0)
    counts = counts.drop(columns=[c for c in counts.columns if c == "CTRL_3"])
    assert "CTRL_3" not in counts.columns and counts.shape[1] == 14, counts.shape
    lg = np.log2(counts / counts.sum(0) * 1e6 + 1)
    z = lg.sub(lg.mean(1), axis=0).div(lg.std(1, ddof=1).replace(0, np.nan), axis=0)
    stage_of = {c: ("E16_5" if c.startswith("E16_5") else c.rsplit("_", 1)[0])
                for c in counts.columns}
    cols = {s: [c for c in counts.columns if stage_of[c] == s] for s in STAGES}
    return z, cols


def panels_bf(repo_root, out_dir):
    """One panel per module: mean z trajectory over sampled per-gene traces."""
    raw_light, _ = _paths(repo_root)
    z, cols = _module_z(raw_light)
    genes = pd.read_csv(os.path.join(raw_light, "Fig3_module_genes.csv"))
    rng = np.random.default_rng(SEED)
    # Assemble every module first so the five panels can share one y range: the point of
    # the row is that the five means peak at different stages with comparable amplitude,
    # which a per-panel autoscale would destroy.
    per_mod = {}
    for m in MODULES:
        g_all = list(genes.loc[genes.module == m, "gene"])
        if len(g_all) != MOD_N[m]:
            raise ValueError(f"{m}: module table has {len(g_all)} genes, canonical is {MOD_N[m]}")
        g = [x for x in g_all if x in z.index]
        S = pd.DataFrame({s: z.loc[g, cols[s]].mean(1) for s in STAGES})
        traces = S.values if len(g) <= N_TRACES else S.values[
            rng.choice(len(g), N_TRACES, replace=False)]
        per_mod[m] = (g_all, g, S, traces, np.nanmean(S.values, axis=0))
    # y limits from the drawn traces, so nothing is clipped off the panel
    allv = np.concatenate([t.ravel() for _, _, _, t, _ in per_mod.values()])
    allv = allv[np.isfinite(allv)]
    lo, hi = float(allv.min()), float(allv.max())
    pad = 0.06 * (hi - lo)
    out, report = {}, []
    for i, m in enumerate(MODULES):
        g_all, g, S, traces, mean = per_mod[m]
        fig, ax = plt.subplots(figsize=(1.95, 1.92))
        x = np.arange(5)
        # grey traces first and rasterized: 350 polylines of live vector are the whole
        # file size of this panel, and they carry texture, not readable values.
        ax.plot(x, traces.T, color="#c9c9c9", lw=0.35, alpha=0.55, zorder=1,
                rasterized=True)
        ax.axhline(0, color="#aaaaaa", lw=0.5, ls=(0, (3, 2)), zorder=2)
        pk = STAGES.index(MOD_PEAK[m])
        ax.axvspan(pk - 0.42, pk + 0.42, color=MOD_COL[m], alpha=0.10, lw=0, zorder=0)
        ax.plot(x, mean, color=MOD_COL[m], lw=2.2, marker="o", ms=5.0, mec="white",
                mew=0.8, zorder=4)
        ax.set_xlim(-0.45, 4.45)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xticks(x)
        ax.set_xticklabels([STAGE_LAB[s] for s in STAGES], rotation=45, ha="right",
                           rotation_mode="anchor", fontsize=6.4, color="#555555")
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.tick_params(labelsize=6.4, length=2.0, pad=1.6)
        ax.set_ylabel("module score (z)", fontsize=7.0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.6)
        ax.set_title(disp(m), fontsize=8.0, color=MOD_COL[m], fontweight="bold", pad=11)
        ax.text(0.5, 1.015, f"(n = {len(g_all)} genes)", transform=ax.transAxes,
                fontsize=6.4, color="#777777", ha="center", va="bottom")
        label = f"A{i + 1}"
        # Filenames follow the internal key, not the display label, so a rename of the
        # fifth program does not break downstream links.
        stem = os.path.join(out_dir, f"Fig3A{i + 1}_trajectory_{m.replace(' ', '_').lower()}")
        render_svg.save_svg(fig, stem)
        plt.close(fig)
        out[label] = stem + ".svg"
        report.append(dict(panel=label, module=m, n_genes_table=len(g_all),
                           n_genes_matched=len(g), n_traces=min(len(g), N_TRACES),
                           peak_recomputed=STAGES[int(np.nanargmax(mean))],
                           peak_expected=MOD_PEAK[m],
                           **{f"z_{s}": round(float(v), 4) for s, v in zip(STAGES, mean)}))
    return out, pd.DataFrame(report)


# panel g: dominant-module UMAPs
def panel_g(repo_root, out_dir):
    """Per-stage UMAP coloured by dominant module.

    dominant-MODULE CALL, z-score each module ACROSS CELLS (axis=1), take the argmax
    over the five modules per cell, and call the cell only when that top z >= 0.5.
    Per-module z is what puts the five scores on a comparable scale: they are sums over
    gene sets of very different size (217 to 985 genes), so the raw scores are not
    mutually rankable and whichever module has the largest gene set would win almost
    everywhere.

    This recipe is verified against the archived panel table
    data/raw_light/Fig3_percell_module_share_by_stage.csv: it reproduces n_called and
    all 25 per-stage module shares exactly (max |difference| 0.0000 at 4 dp).

    The alternative within-cell z (axis=0, over the 5 modules of one cell) is recorded
    here because it was the specification handed to this build, and it is wrong twice
    over: it disagrees with the archived shares by up to 0.46, and its threshold is
    inert, for n=5 values the maximum z-score is >= 1/sqrt(4) = 0.5 by construction, so
    "call at z >= 0.5" would call 100% of cells and the "no dominant module" class could
    never be populated. Observed: 0 of 698,631 cells fall below it. Panel g exports both
    so the discrepancy is auditable, but the per-module recipe is the one drawn.
    """
    _, staged = _paths(repo_root)
    d = np.load(os.path.join(staged, "percell_module_scores.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    if names != MODULES:
        raise ValueError(f"module order in npz is {names}, expected {MODULES}")
    scores = d["scores"]
    umap = d["umap"]
    codes = d["stage_codes"]
    cats = [str(x) for x in d["stage_cats"]]
    # per-module z across cells (axis=1), then argmax over modules per cell
    zm = (scores - scores.mean(axis=1, keepdims=True)) / scores.std(axis=1, keepdims=True)
    assign = zm.argmax(axis=0)
    called = zm.max(axis=0) >= 0.5
    # the rejected within-cell variant, kept only to quantify the disagreement
    zc = (scores - scores.mean(axis=0)) / scores.std(axis=0)
    alt_called = zc.max(axis=0) >= 0.5
    rng = np.random.default_rng(SEED)

    # Explicit margins: the title strip, the per-panel n line, the legend row and the
    # method caption each own a horizontal band. Autolayout stacks them into each other.
    fig, axes = plt.subplots(1, 5, figsize=(11.6, 3.30))
    fig.subplots_adjust(left=0.006, right=0.994, top=0.865, bottom=0.235, wspace=0.02)
    caps = {}
    for ax, st in zip(axes, STAGES):
        code = cats.index(st)
        sel = np.flatnonzero(codes == code)
        n_full = int(sel.size)
        if n_full != STAGE_N[st]:
            raise ValueError(f"{st}: npz has {n_full} cells, canonical is {STAGE_N[st]}")
        other = np.flatnonzero(codes != code)
        # background is COMPLETE, every other-stage cell is drawn, so the reader sees
        # the whole manifold and the stage's occupancy of it, not a thinned silhouette.
        ax.scatter(umap[other, 0], umap[other, 1], s=0.6, c=GREY_OTHER, lw=0,
                   rasterized=True, zorder=1)
        fg = sel if n_full <= FG_CAP else rng.choice(sel, FG_CAP, replace=False)
        caps[st] = int(fg.size)
        unc = fg[~called[fg]]
        if unc.size:
            ax.scatter(umap[unc, 0], umap[unc, 1], s=0.9, c=GREY_UNCALLED, lw=0,
                       rasterized=True, zorder=2)
        cl = fg[called[fg]]
        for mi, m in enumerate(MODULES):
            k = cl[assign[cl] == mi]
            if k.size:
                ax.scatter(umap[k, 0], umap[k, 1], s=0.9, c=MOD_COL[m], lw=0,
                           rasterized=True, zorder=3)
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
        ax.set_title(STAGE_LAB[st], fontsize=8.4, fontweight="bold", pad=14)
        # n rides just under the stage name, inside the panel's own title band, so it
        # cannot reach the legend row below
        ax.text(0.5, 1.005, f"n = {STAGE_N[st]:,}", transform=ax.transAxes,
                fontsize=6.6, color="#666666", ha="center", va="bottom")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", ls="", ms=5, color=MOD_COL[m], label=disp(m))
               for m in MODULES]
    handles += [Line2D([], [], marker="o", ls="", ms=5, color=GREY_UNCALLED,
                       label="no dominant module"),
                Line2D([], [], marker="o", ls="", ms=5, color=GREY_OTHER,
                       label="other stages")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.105),
               ncol=7, frameon=False, fontsize=6.8, handletextpad=0.35,
               columnspacing=1.3)
    fig.text(0.006, 0.988, "Each stage is dominated by its own programme",
             fontsize=9.0, ha="left", va="top")
    capped = [s for s in STAGES if caps[s] < STAGE_N[s]]
    # caption is two lines: one long line overruns the 11.6 in canvas at 6.2 pt
    fig.text(0.006, 0.075,
             "dominant module = the highest of the five scores after z-scoring each module "
             "across cells, called at z \u2265 0.5  \u00b7  n is the stage's full cell count",
             fontsize=6.2, color="#888888", ha="left", va="top")
    fig.text(0.006, 0.035,
             f"foreground points are a fixed-seed display subsample capped at {FG_CAP:,} per "
             f"stage ({', '.join(STAGE_LAB[s] for s in capped)}); the grey background is "
             "every cell from the other four stages, uncapped",
             fontsize=6.2, color="#888888", ha="left", va="top")
    stem = os.path.join(out_dir, "extra_dominant_module_umaps")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    # share of CALLED cells per stage, the quantity the archived table records
    raw_light, _ = _paths(repo_root)
    rows = []
    for st in STAGES:
        m = codes == cats.index(st)
        n_called = int((m & called).sum())
        rows.append(dict(stage=st, n_cells=STAGE_N[st], n_called=n_called,
                         **{mod: round(float(((assign == mi) & called & m).sum())
                                       / max(n_called, 1), 4)
                            for mi, mod in enumerate(MODULES)}))
    frac = pd.DataFrame(rows).set_index("stage")
    ref_path = os.path.join(raw_light, "Fig3_percell_module_share_by_stage.csv")
    check = {}
    if os.path.exists(ref_path):
        ref = pd.read_csv(ref_path).set_index("stage")
        check = dict(
            max_abs_share_diff=float(np.abs(frac[MODULES].values
                                            - ref.loc[frac.index, MODULES].values).max()),
            max_abs_ncalled_diff=int(np.abs(frac.n_called.values
                                            - ref.loc[frac.index, "n_called"].values).max()),
            alt_within_cell_uncalled=int((~alt_called).sum()),
            n_cells_total=int(scores.shape[1]))
        if check["max_abs_share_diff"] > 5e-4 or check["max_abs_ncalled_diff"] != 0:
            raise ValueError(f"dominant-module call disagrees with the archived share "
                             f"table: {check}")
    return stem + ".svg", frac, caps, check


# panel h: the swarm matrix
def panel_h(repo_root, out_dir):
    """16-type x 4-contrast swarm matrix, via build_fig3n_swarm.build() verbatim.

    That module is called through exec() in the original workflow and reads its inputs
    from module-level globals (swarm, sm, pooled, PR_SUMMARY, ROW_ORDER, COMP, STG,
    DENDRO). We inject them rather than editing the builder, it carries shipped
    geometry (the 4+2n gridspec, the measured 2.85-unit label column, the caps and the
    0.94 beeswarm half-height) and must not be touched.
    """
    raw_light, _ = _paths(repo_root)
    sm = pd.read_csv(os.path.join(raw_light, "Fig3n_swarm_DEG_counts.csv"))
    row = pd.read_csv(os.path.join(raw_light, "Fig3n_swarm_row_order.csv"))
    prs = pd.read_csv(os.path.join(raw_light, "Fig3n_pooled_rare_estimates.csv"))
    per = pd.read_csv(os.path.join(raw_light, "Fig3n_percelltype_significant_genes.csv"))
    pool_g = pd.read_csv(os.path.join(raw_light, "Fig3n_pooled_genes_beyond_noise.csv"))
    spine = json.load(open(os.path.join(raw_light, "Fig3n_dendrogram_spine.json")))

    ROW_ORDER = list(row.row_order)
    if spine["leaf_order"] != ROW_ORDER:
        raise ValueError("dendrogram leaf order does not match Fig3n_swarm_row_order.csv")
    COMP = dict(zip(row.row_order, row.compartment))
    STG = [("E16.5", "E16_5"), ("PPD1", "PPD1"), ("PPD7", "PPD7"), ("PPD30", "PPD30")]

    # tested cells: the per-gene log2FC/q table, keyed the way the builder indexes it
    swarm = {(nm, ct): g[["log2FC", "q"]]
             for (contrast, ct), g in per.groupby(["contrast", "ctype"])
             for nm in [contrast.replace("CTRL_vs_", "")]}
    # pooled cells: per-gene pooled log2FC with that gene's OWN noise floor. The floor
    # is per-gene, not the row median, the median only sets the drawn band's width.
    pooled = {(contrast.replace("CTRL_vs_", ""), ct): g[["log2FC", "floor95"]]
              for (contrast, ct), g in pool_g.groupby(["contrast", "ctype"])}
    band = dict(zip(zip(prs.contrast, prs.ctype), prs.median_noise_floor))
    for (nm, ct), g in list(pooled.items()):
        # The archived per-gene table holds only the genes that CLEARED the floor, so a
        # median over it would understate the band. Overwrite floor95 with the archived
        # row-level median so the drawn band is the published sampling-noise floor, and
        # keep every row "beyond", which is exactly what the archived table asserts.
        f = band[(f"CTRL_vs_{nm}", ct)]
        pooled[(nm, ct)] = g.assign(floor95=f)

    # A cell that WAS tested and found nothing has no rows in the per-gene table, which
    # is not the same thing as a cell that could not be tested: it must draw as an empty
    # swarm printing "0", never as a grey not-estimable band. Give it an empty frame.
    tested_zero = []
    for nm, _ in STG:
        for ct in ROW_ORDER:
            rec = sm[(sm.contrast == f"CTRL_vs_{nm}") & (sm.ctype == ct)]
            if rec.empty:
                raise ValueError(f"no count record for ({nm}, {ct})")
            if rec.iloc[0]["mode"] == "tested" and (nm, ct) not in swarm:
                if float(rec.iloc[0].n_sig) != 0:
                    raise ValueError(f"({nm}, {ct}) claims n_sig={rec.iloc[0].n_sig} "
                                     f"but has no per-gene rows")
                swarm[(nm, ct)] = pd.DataFrame({"log2FC": [], "q": []}, dtype=float)
                tested_zero.append((nm, ct))
    missing = [(nm, ct) for nm, _ in STG for ct in ROW_ORDER
               if (nm, ct) not in swarm and (nm, ct) not in pooled]
    if missing:
        raise ValueError(f"no per-gene data for {missing}")

    # The archived counts table calls a pooled row mode="pooled"; the builder branches on
    # mode=="not_testable" (its vocabulary: not testable at the replicate level, hence
    # pooled). Translate at injection so the builder stays byte-identical, and keep the
    # archived table on disk unchanged.
    sm_b = sm.copy()
    sm_b.loc[sm_b["mode"] == "pooled", "mode"] = "not_testable"

    # inject into the verbatim builder's namespace, then call build() unmodified
    swarm_mod.swarm = swarm
    swarm_mod.sm = sm_b
    swarm_mod.pooled = pooled
    swarm_mod.PR_SUMMARY = prs
    swarm_mod.ROW_ORDER = ROW_ORDER
    swarm_mod.COMP = COMP
    swarm_mod.STG = STG
    swarm_mod.DENDRO = {"dcoord": spine["dcoord"], "icoord": spine["icoord"]}
    fig, n_markers = swarm_mod.build(cap_ns=140, cap_sig=260, cap_pooled=110,
                                     figsize=(33.2, 18.72), fs_title=30, fs_lab=24,
                                     fs_small=20, s_point=5.5)
    stem = os.path.join(out_dir, "extra_swarm_matrix_4stage")
    render_svg.save_svg(fig, stem)
    plt.close(fig)
    return stem + ".svg", int(n_markers), sm, tested_zero


# driver
def build(repo_root: str, out_dir: str) -> dict:
    """Render every Fig3 panel into out_dir. Returns {panel_label: svg_path}."""
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    out = {}
    out["a"] = panel_a(out_dir)
    bf, traj_report = panels_bf(repo_root, out_dir)
    out.update(bf)
    out["g"], frac, caps, gcheck = panel_g(repo_root, out_dir)
    out["h"], n_markers, sm, tested_zero = panel_h(repo_root, out_dir)
    build.report = dict(trajectories=traj_report, dominant_share=frac,
                        dominant_check=gcheck, foreground_caps=caps,
                        swarm_markers=n_markers, tested_zero=tested_zero,
                        swarm_counts=sm)
    return out


if __name__ == "__main__":
    import sys
    r = sys.argv[1]
    print(build(r, os.path.join(r, "figures")))
