# Inputs

Two classes of input feed the notebooks.

## 1. `data/raw_light/`: tables and images (tracked)

Analysis results and measurements: program definitions, DEG counts, GO selections,
pseudobulk matrices, the LIANA output restricted to the epithelium as target, the
section morphometry, the tracked organoid diameters and the four representative well
images. `MANIFEST.csv` records, per file, the original path, size, sha256 and shape.
The notebooks read these files; they do not recompute them.

Files that are not straight copies:

- `Fig2d_deck_marker_order.json`: the 139-gene column order of the Fig. 2D marker
  heatmap, read positionally from the rendered panel (the x coordinate of each
  rotated `<text>` element). PDF text extraction returns reading order, not visual
  column order, and must not be used for this. Rows of that panel are alphabetical.
  The matrix itself is `data/staged/marker_zscore_deck139.csv`, written by
  `scripts/stage_marker_deck139.py` from `pseudobulk_rep_by_type.npz`.
- `liana/liana_16types_to_epithelium.csv`: the rows of the 16-type LIANA run with
  `target == Epithelial`, keeping source, ligand, receptor, stage, `lr_means` and
  `cellphone_pvals` (35,946 rows). The full run (449,614 rows) is deposited with the
  data.
- `organoids/per_organoid_humanscored.csv`: one row per tracked organoid that both
  raters scored INCLUDE, with its equivalent diameter at 24, 48, 72 and 96 h (2.50
  um per pixel in the projections; see `organoids/CALIBRATION.txt`). Tracking code
  and per-well detections are in the data deposit, not in this repository.
- `organoids/images/<well>_<hour>h.png`: the whole-well maximum projections of the
  four wells shown in Fig. 5 D and Fig. 6 C (B3 vehicle, B6 TNFa, D3 Matrigel only,
  E2 10 % collagen I).

## 2. `data/staged/`: derived tables (tracked, regenerable)

Four files are written by `scripts/00_stage_inputs.py` in one streaming pass over
`endometrium_annotated_v1.h5ad` (14.3 GB; 698,631 x 17,987 CSR int64 counts); the
script reads X in 25,000-cell row blocks because the file cannot be loaded whole on an
8 GB machine. The fifth, `marker_zscore_deck139.csv`, is written by
`scripts/stage_marker_deck139.py` from `raw_light/pseudobulk_rep_by_type.npz` and
needs no atlas.

| file | shape | feeds |
|---|---|---|
| `cells.parquet` | 698,631 x 9 | Fig. 2 A, B, C, E; fig. S2 E |
| `composition_by_replicate.csv` | 15 replicates x 16 types | Fig. 2 F |
| `percell_module_scores.npz` | scores (5, 698,631) | Fig. 3 A, extras |
| `marker_zscore.csv` | 16 x 66 | kept for completeness; not drawn |
| `marker_zscore_deck139.csv` | 16 x 139 | Fig. 2 D (from pseudobulk, `stage_marker_deck139.py`) |

The script checks the atlas shape, the five per-stage cell counts, 16 cell types, 4
compartments and the absence of NaN, and exits non-zero on any mismatch.

## Definitions that are read, not re-derived

- Cell-type names come from `group_map_17.csv` keyed on `obs['RNA_snn_res.0.2']`
  (22 clusters), not from `obs['cluster_label']`, which holds provisional strings
  (cluster 20 is Smooth muscle).
- The five whole-tissue programs (Homeostasis 678, Pregnancy 217, Early repair 985,
  Late repair 745, Remodelling 789 genes) are the gene lists in
  `Fig3_module_genes.csv`.
- The fifth program is displayed as "New Homeostasis"; its key in every table is
  `Remodelling` and the rename is applied at display time by `disp()`.
- `CTRL_3` has 36 cells (QC fail). It is flagged in `composition_by_replicate.csv`
  and excluded from replicate-level pseudobulk.
- Program score per cell: counts -> counts per 10,000 -> log1p, then the mean over
  the program's genes with zeros included (not `scanpy.tl.score_genes`). The 1e4
  target is part of the definition; at 1e4 the regenerated scores match the shipped
  reference at r = 1.000 per program.

## Builders in `src/`

`build_fig3n_swarm.py`, `build_fig4_traj.py`, `build_fig4_heatmap.py` and
`build_fig4_go.py` carry the panel geometry of the heatmap, GO and swarm panels and
are called by `programs_whole_tissue.py` and `programs_by_compartment.py`.
`build_fig4_heatmap.build()` updates `mpl.rcParams` internally, which resets
`svg.fonttype` to `path`; it accepts a `save=` callback and the caller passes
`render_svg.save_svg` so the export keeps live text.
