"""Figure 5. Collagen I input to the epithelium and organoid growth in 10 % collagen I.

  A schematic of the organoid dome (drawing, not computed)
  B ligands present at every stage ranked by mean input to the epithelium
  C collagen I senders per stage, chord circles on one ribbon scale
  D representative wells, Matrigel only vs 10 % collagen I, 24-96 h
  E organoid diameter per timepoint
  F diameter kinetics with the mixed-model star per timepoint

Entry point: build(repo_root, out_dir) -> {panel_label: path}
"""
import os

import liana_panels
import organoid_panels

CTRL, FAC = "Matrigel only", "Collagen I"


def build(repo_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = {"A": "SCHEMATIC: organoid dome in 10 % collagen I, drawn in Illustrator"}
    li = liana_panels.build(repo_root, out_dir)
    out["B"], out["C"] = li["5B"], li["5C"]
    col = organoid_panels.COLOUR[FAC]
    wells = organoid_panels.build_wells(repo_root, out_dir, CTRL, FAC, col,
                                       os.path.join(out_dir, "Fig5D_wells_CollagenI"))
    out["D"] = wells["wells"]
    pair = organoid_panels.build_pair(repo_root, out_dir, CTRL, FAC, col, "Fig5E_swarm_CollagenI",
                                      "Fig5F_kinetics_CollagenI", ctrl_label="Matrigel")
    out["E"], out["F"] = pair["swarm"], pair["kinetics"]
    build.report = dict(top_generic=li["tables"]["top_generic"], picks=wells["picks"], stats=pair["stats"])
    return out
