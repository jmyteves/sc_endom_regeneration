"""Fig. S1. Dissociation optimisation (30 / 45 / 60 min digestion).

None of the four panels can be rebuilt from this repository. The series exists
as raw FCS files (13 tubes, 7-AAD, EpCAM BV510, PDGFRa PE-CF594, CD45 FITC,
CD31 PE-Cy7); the gate statistics were never exported and no file records the
tissue mass behind the cells-per-mg panel. build() returns a reason string per
panel and writes a placeholder slug beside it. The values printed on the
delivered panels are quoted in the reasons so they can be checked against the
figure.

Entry point: build(repo_root, out_dir) -> {panel_label: reason}
"""
import os

import render_svg
from library_qc_panels import _slug

NOT_COMPUTABLE = {
    "A": "flow density plots, 30 / 45 / 60 min digestion x 7-AAD, EpCAM, PDGFRa, "
         "CD45, CD31: raw FCS only, no gate export",
    "B": "viability as percent of singlets by digestion time (delivered 81.0 / 56.4 "
         "/ 60.0): no gate statistics on disk",
    "C": "total live cells per mg by digestion time (delivered 5.9 / 12.7 / 3.0): no "
         "FCS keyword or table records a tissue mass",
    "D": "percent of live cells positive per marker by digestion time: needs the "
         "live gate, which was never exported",
}


def build(repo_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    render_svg.svg_mode()
    out = {}
    for lab, reason in NOT_COMPUTABLE.items():
        _slug(lab, reason, out_dir)
        out[lab] = f"NOT COMPUTABLE: {reason}"
    return out
