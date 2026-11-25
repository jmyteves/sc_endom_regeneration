# Panel map

Deck v10 panel -> status -> module -> file in `figures/`. The same table is `panel_map.csv`.

| figure | panel | content | status | module | output |
|---|---|---|---|---|---|
| Fig. 1 | A | timeline schematic | builds | fig1_panels | Fig1A_timeline_schematic.svg |
| Fig. 1 | B | uterine horn length | builds | fig1_panels | Fig1B_horn_length.svg |
| Fig. 1 | C | Tukey HSD matrix, 5 stages | builds | fig1_panels | Fig1C_tukey_matrix.svg |
| Fig. 1 | D | gross photographs | wet-lab placeholder | fig1_panels | Fig1D_WETLAB_placeholder.svg |
| Fig. 1 | E | EPCAM / FOXA2 immunofluorescence | wet-lab placeholder | fig1_panels | Fig1E_WETLAB_placeholder.svg |
| Fig. 1 | F | luminal epithelium : stroma area ratio | builds (new) | fig1_panels | Fig1F_LE_stroma_ratio.svg; Fig1FGH_morphometry_strip.svg |
| Fig. 1 | G | endometrium / uterus area ratio | builds (new) | fig1_panels | Fig1G_E_fraction.svg; Fig1FGH_morphometry_strip.svg |
| Fig. 1 | H | glands per cross-section | builds (new) | fig1_panels | Fig1H_gland_count_mean.svg; Fig1FGH_morphometry_strip.svg |
| Fig. 2 | A | UMAP by stage | builds | fig2_panels | Fig2A_umap_stage.svg |
| Fig. 2 | B | UMAP by compartment | builds | fig2_panels | Fig2B_umap_compartment.svg |
| Fig. 2 | C | UMAP by cell identity | builds | fig2_panels | Fig2C_umap_celltype.svg |
| Fig. 2 | D | marker heatmap | builds | fig2_panels | Fig2D_marker_heatmap.svg |
| Fig. 2 | E | UMAP per stage | builds | fig2_panels | Fig2E_umap_per_stage.svg |
| Fig. 2 | F | composition bars | builds | fig2_panels | Fig2F_composition_bars.svg |
| Fig. 3 | A | five program trajectories | builds (re-lettered from old 3 b-f) | fig3_panels / programs_whole_tissue | Fig3A1-5_trajectory_*.svg |
| Fig. 3 | B | program signature heatmap | builds (re-lettered from old 4f) | fig3_panels / programs_by_compartment | Fig3B_program_signature_heatmap.svg |
| Fig. 3 | C | program GO dotplot | builds (re-lettered from old 4g) | fig3_panels / programs_by_compartment | Fig3C_program_GO_dotplot.svg |
| Fig. 3 | D | Masson's trichrome sections | wet-lab placeholder | fig3_panels | (reason string) |
| Fig. 4 | A | program trajectories by compartment | builds (re-lettered from old 4 a-e) | fig4_panels / programs_by_compartment | Fig4A_module_trajectories_by_compartment.svg |
| Fig. 4 | B-E | k = 5 modules + New Homeostasis GO per compartment | builds (re-lettered from old Fig. 5) | fig4_panels / k5_modules_sheet | Fig4BE_k5_modules_GO_A4.svg |
| Fig. 5 | A | organoid schematic | schematic (Illustrator) | fig5_panels | (reason string) |
| Fig. 5 | B | top 10 ligands, all stages | builds (new) | fig5_panels / liana_panels | Fig5B_collagenI_top_generic_ligand.svg |
| Fig. 5 | C | collagen I chord circles per stage | builds (new) | fig5_panels / liana_panels | Fig5C_collagenI_chords_per_stage.svg |
| Fig. 5 | D | wells, Matrigel only vs 10 % collagen I | builds (new) | fig5_panels / organoid_panels | Fig5D_wells_CollagenI.svg |
| Fig. 5 | E | diameter swarm | builds (new) | fig5_panels / organoid_panels | Fig5E_swarm_CollagenI.svg |
| Fig. 5 | F | diameter kinetics | builds (new) | fig5_panels / organoid_panels | Fig5F_kinetics_CollagenI.svg |
| Fig. 6 | A | top 10 ligands acquired at PPD30 | builds (new) | fig6_panels / liana_panels | Fig6A_TNFa_top_acquired_PPD30.svg |
| Fig. 6 | B | TNFa chord circle + legend | builds (new) | fig6_panels / liana_panels | Fig6B_TNFa_chord_increase_PPD30.svg; Fig6B_sender_legend.svg |
| Fig. 6 | C | wells, vehicle vs TNFa | builds (new) | fig6_panels / organoid_panels | Fig6C_wells_TNFa.svg |
| Fig. 6 | D | diameter swarm | builds (new) | fig6_panels / organoid_panels | Fig6D_swarm_TNFa.svg |
| Fig. 6 | E | diameter kinetics | builds (new) | fig6_panels / organoid_panels | Fig6E_kinetics_TNFa.svg |
| fig. S1 | A-D | digestion series: flow plots, viability, cells/mg, % positive | NOT COMPUTABLE (raw FCS only) | sfig1_panels | SFig1A-D_NOTCOMPUTABLE_placeholder.svg |
| fig. S2 | A | cells passing QC per library | builds (old S1 i) | sfig2_panels / library_qc_panels | SFig2A_cells_passing_qc.svg |
| fig. S2 | B | count-feature fit | builds (old S1 k) | sfig2_panels / library_qc_panels | SFig2B_count_feature_fit.svg |
| fig. S2 | C | cells per stage | builds (new) | sfig2_panels | SFig2C_cells_per_stage.svg |
| fig. S2 | D | QC medians per stage | builds (old S1 j) | sfig2_panels / library_qc_panels | SFig2D_qc_distributions.svg |
| fig. S2 | E | Harmony vs raw-PCA UMAPs | Harmony half builds (old S2 d); raw-PCA half NOT COMPUTABLE | sfig2_panels / atlas_validity_panels | SFig2E_umap_harmony.svg; SFig2E_rawPCA_NOTCOMPUTABLE_placeholder.svg |
| fig. S2 | F | MCA concordance heatmap | NOT COMPUTABLE (raster only) | sfig2_panels | SFig2F_NOTCOMPUTABLE_placeholder.svg |
| fig. S2 | G | cells per replicate | builds (old S2 f) | sfig2_panels / atlas_validity_panels | SFig2G_cells_per_replicate.svg |
| fig. S2 | H | within-replicate composition | builds (old S2 g) | sfig2_panels / atlas_validity_panels | SFig2H_composition_per_replicate.svg |
| fig. S3 | A | volcanoes | builds | sfig3_panels / program_robustness_panels | SFig3A_volcanoes_ctrl_anchored.svg |
| fig. S3 | B | program null bands | builds | sfig3_panels / program_robustness_panels | SFig3B_program_null_bands.svg |
| fig. S3 | C | program overlap | builds | sfig3_panels / program_robustness_panels | SFig3C_program_overlap.svg |
| fig. S3 | D | program heatmaps by compartment | builds | sfig3_panels / program_robustness_panels | SFig3D_program_heatmaps_by_compartment.svg |
| fig. S4 | A | GO matrix, 4 compartments | builds (old S3 e) | sfig4_panels / program_robustness_panels | SFig4A_program_GO_matrix_4compartments.svg |
| fig. S5 | A-D | LR dot plots, principal pairs | builds (old S4 c) | sfig5_panels / signalling_controls_panels | SFig5AD_LR_dotplots_principal_pairs.svg |
| extra |  | five-wave schematic (old Fig. 3 a) | built, not in deck | extras_panels / programs_whole_tissue | extra_five_wave_schematic.svg |
| extra |  | dominant-program UMAPs (old Fig. 3 g) | built, not in deck | extras_panels / programs_whole_tissue | extra_dominant_module_umaps.svg |
| extra |  | DEG swarm matrix (old Fig. 3 h) | built, not in deck | extras_panels / programs_whole_tissue | extra_swarm_matrix_4stage.svg |
| extra |  | 16-type signalling composite (old Fig. 6) | built, not in deck | extras_panels / signalling_16types | extra_signalling_vs_control_16types.svg |
| extra |  | flow gate statistics, doublet summary (old S1 g, h) | built, not in deck | extras_panels / library_qc_panels | extra_flow_gate_statistics.svg; extra_doublet_summary.svg |
| extra |  | marker grid, QC UMAP overlays, equal-N subsampling (old S2 a, e, h) | built, not in deck | extras_panels / atlas_validity_panels | extra_marker_grid.svg; extra_qc_umap_overlays.svg; extra_equalN_subsampling.svg |
| extra |  | interactions vs depth, coverage, epithelial LR share, rewiring (old S4 a, b, d, e) | built, not in deck | extras_panels / signalling_controls_panels | extra_interactions_vs_depth.svg; extra_coverage_matrix.svg; extra_epithelial_LR_share.svg; extra_rewiring_vs_control.svg |
| extra |  | coordination and layer retention (old S5 a-d) | built, not in deck | extras_panels / coordination_panels | extra_coordination_*.svg; extra_between_stage_similarity.svg; extra_cross_compartment_matrices.svg; extra_layer_retention_by_anchor.svg |

## Inputs per module

| module | reads |
|---|---|
| fig1_panels | raw_light/horn_length_measurements.csv; raw_light/fig1/morphometry_complete.csv |
| fig2_panels | staged/cells.parquet; staged/composition_by_replicate.csv; staged/marker_zscore_deck139.csv; raw_light/palette_final.json; raw_light/Fig2d_deck_marker_order.json |
| programs_whole_tissue | raw_light/MODULES_summary.csv; raw_light/Fig3_module_genes.csv; raw_light/Fig3n_*; raw_light/Fig7_wholeTissue_persample_pseudobulk_counts.csv; staged/percell_module_scores.npz |
| programs_by_compartment | raw_light/Fig4_*; raw_light/Fig3_heatmap_column_annotation.csv; raw_light/Fig3_column_state_calls.csv; raw_light/pseudobulk_rep_by_type.npz |
| k5_modules_sheet | raw_light/fig5_k5/* |
| liana_panels | raw_light/liana/liana_16types_to_epithelium.csv |
| organoid_panels | raw_light/organoids/per_organoid_humanscored.csv; representative_organoids.csv; tracks/*; images/* |
| library_qc_panels | raw_light/sfig1/* |
| atlas_validity_panels | staged/cells.parquet; staged/marker_zscore_deck139.csv; raw_light/sfig2/*; raw_light/palette_final.json |
| program_robustness_panels | raw_light/sfig3/*; raw_light/Fig7_wholeTissue_persample_pseudobulk_counts.csv |
| signalling_controls_panels | raw_light/sfig4/* |
| signalling_16types | raw_light/fig6/*; raw_light/palette_final.json |
| coordination_panels | raw_light/sfig5/* |
