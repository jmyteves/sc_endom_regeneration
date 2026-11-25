"""Figure 3. Five transcriptional programs peak sequentially.

  A the five whole-tissue program trajectories (one file per program, stacked
     in the figure): coloured mean z over sampled per-gene traces
  B program signature heatmap, cell-type profiles by stage
  C program GO dotplot
  D Masson's trichrome whole sections, one per stage (microscopy, not computed)

Entry point: build(repo_root, out_dir) -> {panel_label: path}
"""
import os

import render_svg
import programs_by_compartment
import programs_whole_tissue

WETLAB = {"D": "WET-LAB: Masson's trichrome whole cross-sections, one per stage"}


def build(repo_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    out, traj_report = programs_whole_tissue.panels_bf(repo_root, out_dir)
    out["B"], hinfo = programs_by_compartment.panel_f(repo_root, out_dir)
    out["C"], ginfo = programs_by_compartment.panel_g(repo_root, out_dir)
    out.update(WETLAB)
    build.report = dict(trajectories=traj_report, heatmap=hinfo, go=ginfo)
    return out
