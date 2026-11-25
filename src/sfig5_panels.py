"""Fig. S5. Ligand-receptor detail of the epithelium (principal pairs).

  A-D dot plots of the principal ligand-receptor pairs, stromal -> epithelial,
       epithelial -> stromal, immune -> epithelial, epithelial -> immune; colour
       is the control share of the pair's signal, dot size the total signal.
       The four blocks are drawn on one sheet.

Entry point: build(repo_root, out_dir) -> {panel_label: path}
"""
import signalling_controls_panels as sc


def build(repo_root, out_dir):
    out = sc.build(repo_root, out_dir, letters=["c"])
    return {k: out["c"] for k in "ABCD"}
