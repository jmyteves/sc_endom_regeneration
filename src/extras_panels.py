"""Panels built for earlier versions of the figure set that are not in the v10 deck.

They still build from the repository data and are kept for reference and for
reviewer requests. Every file is prefixed `extra_`.

  five-wave schematic, dominant-program UMAPs, 16-type DEG swarm matrix
      (former Fig. 3 a, g, h)
  signalling-vs-control composite over 16 cell types (former Fig. 6)
  flow gate statistics and doublet summary (former fig. S1 g, h)
  marker grid, QC UMAP overlays, equal-N subsampling (former fig. S2 a, e, h)
  interactions vs depth, coverage matrix, epithelial LR share, rewiring vs
      control (former fig. S4 a, b, d, e)
  coordination and layer-retention panels (former fig. S5 a-d)

Entry point: build(repo_root, out_dir) -> {label: path}
"""
import os

import render_svg
import atlas_validity_panels as av
import coordination_panels
import library_qc_panels as lq
import programs_whole_tissue as pw
import signalling_16types
import signalling_controls_panels as sc


def build(repo_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    out = {}
    out["five_wave_schematic"] = pw.panel_a(out_dir)
    out["dominant_module_umaps"] = pw.panel_g(repo_root, out_dir)[0]
    out["swarm_matrix_4stage"] = pw.panel_h(repo_root, out_dir)[0]
    out["signalling_vs_control_16types"] = signalling_16types.build(repo_root, out_dir)["a"]
    q = lq.load_inputs(repo_root)
    render_svg.svg_mode()
    out["flow_gate_statistics"] = lq.panel_g(q, out_dir)
    out["doublet_summary"] = lq.panel_h(q["doublet"], out_dir)
    a = av.load_inputs(repo_root)
    out["marker_grid"] = av.panel_a(a["mz"], out_dir)
    out["qc_umap_overlays"] = av.panel_e(a["cells"], out_dir)
    out["equalN_subsampling"] = av.panel_h(a["equal_n"], a["boot"], a["stats"], out_dir)
    for k, v in sc.build(repo_root, out_dir, letters=["a", "b", "d", "e"]).items():
        out[f"signalling_control_{k}"] = v
    for k, v in coordination_panels.build(repo_root, out_dir).items():
        out[f"coordination_{k}"] = v
    return out
