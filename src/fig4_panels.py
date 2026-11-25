"""Figure 4. Programs resolved per compartment.

  A per-compartment program trajectories, one small panel per program
  B-E per-compartment k = 5 pseudobulk modules (heatmap) with the GO terms of
       each compartment's New Homeostasis module below; one A4 sheet holds all
       four columns

Entry point: build(repo_root, out_dir) -> {panel_label: path}
"""
import os

import render_svg
import k5_modules_sheet
import programs_by_compartment


def build(repo_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    out, conc = programs_by_compartment.panels_ae(repo_root, out_dir)
    out = {"A": out["a"]}
    sheet = k5_modules_sheet.build(repo_root, out_dir)
    for k in "BCDE":
        out[k] = sheet["a"]
    build.report = dict(peak_concordance=conc, k5=k5_modules_sheet.build.report)
    return out
