"""Figure 6. TNFa, an epithelial input acquired at PPD30, and organoid growth with TNFa.

  A ligands absent in control ranked by their PPD30 input to the epithelium
  B senders whose TNFa input rose from control to PPD30 (chord circle + legend)
  C representative wells, vehicle vs 10 ng/mL TNFa, 24-96 h
  D organoid diameter per timepoint
  E diameter kinetics with the mixed-model star per timepoint

Entry point: build(repo_root, out_dir) -> {panel_label: path}
"""
import os

import liana_panels
import organoid_panels

CTRL, FAC = "Vehicle", "TNFα"


def build(repo_root, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    out = {}
    li = liana_panels.build(repo_root, out_dir)
    out["A"], out["B"], out["B_legend"] = li["6A"], li["6B"], li["6B_legend"]
    col = organoid_panels.COLOUR[FAC]
    wells = organoid_panels.build_wells(repo_root, out_dir, CTRL, FAC, col,
                                       os.path.join(out_dir, "Fig6C_wells_TNFa"))
    out["C"] = wells["wells"]
    pair = organoid_panels.build_pair(repo_root, out_dir, CTRL, FAC, col, "Fig6D_swarm_TNFa",
                                      "Fig6E_kinetics_TNFa", title="TNFα: larger only at 72 h")
    out["D"], out["E"] = pair["swarm"], pair["kinetics"]
    build.report = dict(top_acquired=li["tables"]["top_acquired"], picks=wells["picks"], stats=pair["stats"])
    return out
