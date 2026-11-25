#!/usr/bin/env python
"""Write the two figure notebooks from the figure-module table below.

    python scripts/build_notebooks.py

Each notebook is a header, a setup cell that locates the repository root, then one
markdown + one code cell per figure. The code cell calls the module's build() and
lists what it wrote. Editing a panel means editing src/, then re-running the
notebook; this script only needs re-running when a figure is added or removed.
"""
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SETUP = '''import os
import sys


def find_repo_root():
    """Walk up from the working directory until src/, data/ and scripts/ are all present."""
    if os.environ.get("SC_ENDOM_ROOT"):
        return os.path.abspath(os.environ["SC_ENDOM_ROOT"])
    d = os.path.abspath(os.getcwd())
    while True:
        if all(os.path.isdir(os.path.join(d, k)) for k in ("src", "data", "scripts")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            raise RuntimeError("repository root not found; set SC_ENDOM_ROOT")
        d = parent


REPO_ROOT = find_repo_root()
FIGS = os.path.join(REPO_ROOT, "figures")
os.makedirs(FIGS, exist_ok=True)
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))
print("repository:", REPO_ROOT)
print("figures   :", FIGS)
'''

CELL = '''import {mod}

out = {mod}.build(REPO_ROOT, FIGS)
for label, path in out.items():
    if isinstance(path, str) and os.path.exists(path):
        print(f"{{label:>10}}  {{os.path.relpath(path, REPO_ROOT)}}")
    elif isinstance(path, str):
        print(f"{{label:>10}}  {{path}}")
'''

CHECK = '''import glob

import render_svg

svgs = sorted(glob.glob(os.path.join(FIGS, "{pattern}")))
bad = []
for p in svgs:
    v = render_svg.verify(p)
    if v["text_elements"] == 0 or not v["named_font"]:
        bad.append(v)
print(len(svgs), "SVG files;", len(bad), "without live text")
for v in bad:
    print(v)
'''

MAIN = [
    ("fig1_panels", "Figure 1. Gross involution and epithelial continuity",
     "A timeline, B horn length, C Tukey matrix, F luminal epithelium : stroma, G endometrium / "
     "uterus area, H glands per cross-section. D and E are microscopy and return a placeholder."),
    ("fig2_panels", "Figure 2. Whole-uterus single-cell atlas",
     "A-C UMAPs by stage, compartment and cell identity, D marker heatmap, E per-stage UMAPs, "
     "F composition bars."),
    ("fig3_panels", "Figure 3. Five transcriptional programs",
     "A program trajectories (one file per program), B program signature heatmap, C program GO "
     "dotplot. D (Masson's trichrome) is microscopy."),
    ("fig4_panels", "Figure 4. Programs per compartment",
     "A per-compartment program trajectories; B-E the k = 5 module heatmaps with the New "
     "Homeostasis GO terms, one A4 sheet."),
    ("fig5_panels", "Figure 5. Collagen I and organoid growth in 10 % collagen I",
     "B ligand ranking across stages, C collagen I chord circles per stage, D representative "
     "wells, E diameter swarm, F diameter kinetics. A is a schematic."),
    ("fig6_panels", "Figure 6. TNFa and organoid growth with TNFa",
     "A ligands acquired at PPD30, B TNFa senders (chord circle and legend), C representative "
     "wells, D diameter swarm, E diameter kinetics."),
]

SUPP = [
    ("sfig1_panels", "Fig. S1. Dissociation optimisation",
     "Not reproducible from this repository (raw FCS only); placeholders with the reason."),
    ("sfig2_panels", "Fig. S2. Annotation validity, batch integration, clustering robustness",
     "A cells per library, B count-feature fit, C cells per stage, D QC medians, E Harmony UMAP "
     "(raw-PCA half is a placeholder), F MCA concordance (placeholder), G cells per replicate, "
     "H within-replicate composition."),
    ("sfig3_panels", "Fig. S3. Program robustness",
     "A volcanoes, B null bands, C program overlap, D program heatmaps by compartment."),
    ("sfig4_panels", "Fig. S4. Program GO terms across compartments",
     "A GO matrix of the 30 Fig. 3C terms against the 20 compartment modules."),
    ("sfig5_panels", "Fig. S5. Epithelial ligand-receptor detail",
     "A-D principal ligand-receptor pairs between epithelium and stroma / immune cells, both "
     "directions, on one sheet."),
    ("extras_panels", "Panels not in the v10 figure set",
     "Earlier panels kept for reference (prefix extra_): five-wave schematic, dominant-program "
     "UMAPs, DEG swarm matrix, the 16-type signalling composite, flow gate statistics, doublet "
     "summary, marker grid, QC UMAP overlays, equal-N subsampling, signalling controls, "
     "coordination and layer retention."),
]


def notebook(title, intro, figures, pattern):
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    cells = [nbf.v4.new_markdown_cell(f"# {title}\n\n{intro}"), nbf.v4.new_code_cell(SETUP)]
    for mod, head, desc in figures:
        cells.append(nbf.v4.new_markdown_cell(f"## {head}\n\n{desc}\n\nCode: `src/{mod}.py`"))
        cells.append(nbf.v4.new_code_cell(CELL.format(mod=mod)))
    cells.append(nbf.v4.new_markdown_cell("## Check\n\nEvery SVG must carry live text in a named font."))
    cells.append(nbf.v4.new_code_cell(CHECK.format(pattern=pattern)))
    nb.cells = cells
    return nb


def main():
    out = os.path.join(ROOT, "notebooks")
    os.makedirs(out, exist_ok=True)
    intro_main = ("Mouse uterus scRNA-seq time course (Control, E16.5, PPD1, PPD7, PPD30; 698,631 cells) "
                  "with histology, morphometry and organoid assays. Each cell below rebuilds one main "
                  "figure from `data/` into `figures/`. Nothing here opens the atlas h5ad; "
                  "`data/staged/` already holds everything derived from it.")
    intro_supp = ("Supplementary figures S1-S5 of the same manuscript, plus the panels that were built "
                  "for earlier versions and are not in the current figure set.")
    nbf.write(notebook("sc_endom main figures", intro_main, MAIN, "Fig*.svg"),
              os.path.join(out, "sc_endom_figures.ipynb"))
    nbf.write(notebook("sc_endom supplementary figures", intro_supp, SUPP, "SFig*.svg"),
              os.path.join(out, "sc_endom_supplementary.ipynb"))
    print("wrote notebooks/sc_endom_figures.ipynb and notebooks/sc_endom_supplementary.ipynb")


if __name__ == "__main__":
    main()
