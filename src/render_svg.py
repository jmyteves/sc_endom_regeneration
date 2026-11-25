"""SVG export helpers shared by every figure module.

Conventions
  - svg.fonttype = 'none' -> text stays as <text> elements, editable in Illustrator
  - named font stack, Arial first
  - transparent figure and axes patches
  - .svg is the deliverable; .pdf is written alongside for preview

Call `svg_mode()` (or `arial_mode()` for the slot-sized panels) before building a
figure, then `save_svg(fig, stem)` or `save_fixed(fig, stem)`.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt

FONT_STACK = ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"]


def svg_mode(base=8, small=7, tiny=6):
    """Editable-text, transparent-background rcParams."""
    mpl.rcParams.update({
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": FONT_STACK,
        "figure.facecolor": "none",
        "axes.facecolor": "none",
        "savefig.facecolor": "none",
        "savefig.edgecolor": "none",
        "savefig.transparent": True,
        "font.size": small,
        "axes.titlesize": base,
        "axes.labelsize": small,
        "xtick.labelsize": tiny,
        "ytick.labelsize": tiny,
        "legend.fontsize": tiny,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.linewidth": 0.8,
        "legend.frameon": False,
        "figure.dpi": 110,
    })


def arial_mode(base=7, secondary=6, tick=6):
    """rcParams for the slot-sized panels placed at 100 % in Illustrator.

    Arial, live text, 0.6 pt axes, no top/right spines. Figure size is set per
    panel, so savefig.bbox stays None and the declared figsize is kept.
    """
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": base,
        "axes.labelsize": base,
        "axes.titlesize": base,
        "legend.fontsize": secondary,
        "xtick.labelsize": tick,
        "ytick.labelsize": tick,
        "axes.linewidth": 0.6,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.size": 3, "ytick.major.size": 3,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "normal",
        "axes.titlelocation": "left",
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "patch.linewidth": 0.6,
        "svg.fonttype": "none",
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": None,
        "figure.dpi": 110,
    })


def save_fixed(fig, stem, png_dpi=300, also_pdf=True, png=True):
    """Write <stem>.svg (+ .pdf, + preview .png) at the declared figsize.

    Unlike save_svg this does not crop to content: the slot-sized panels must
    come out at exactly the figsize they were drawn at.
    """
    with mpl.rc_context({"savefig.bbox": None}):
        fig.savefig(stem + ".svg", transparent=True)
        if also_pdf:
            fig.savefig(stem + ".pdf", transparent=True)
        if png:
            fig.savefig(stem + "_preview.png", dpi=png_dpi, facecolor="white")
    return stem + ".svg"


def transparent(fig):
    """Strip every opaque patch so the SVG composites over any background."""
    fig.patch.set_alpha(0.0)
    fig.patch.set_facecolor("none")
    for ax in fig.axes:
        ax.patch.set_alpha(0.0)
        ax.patch.set_facecolor("none")
        leg = ax.get_legend()
        if leg is not None:
            leg.get_frame().set_alpha(0.0)
            leg.get_frame().set_facecolor("none")
    return fig


def check_text(fig):
    """Report horizontal text-bbox collisions (rotated text excluded).

    Draws first: legend and annotation extents are unset until the figure is
    rendered once, which otherwise reports every legend entry at (0,0) as
    mutually overlapping.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tx = [(t, t.get_window_extent(r)) for t in fig.findobj(mpl.text.Text)
          if t.get_text().strip() and t.get_visible() and t.get_rotation() == 0]
    return [(a.get_text()[:14], b.get_text()[:14])
            for i, (a, ba) in enumerate(tx) for b, bb in tx[i + 1:] if ba.overlaps(bb)]


def save_svg(fig, stem, also_pdf=True):
    """Write <stem>.svg (+ .pdf), transparent, with live text. Returns collisions."""
    transparent(fig)
    ov = check_text(fig)
    fig.savefig(stem + ".svg", format="svg", bbox_inches="tight", transparent=True)
    if also_pdf:
        fig.savefig(stem + ".pdf", bbox_inches="tight", transparent=True)
    return ov


def verify(path, expect_words=()):
    """Confirm the SVG has live <text> (not <path> outlines) and no opaque backdrop."""
    s = open(path, encoding="utf-8").read()
    n_text = s.count("<text")
    named = any(f in s for f in FONT_STACK)
    white_rect = 'style="fill: #ffffff"' in s or 'fill:#ffffff' in s.replace(" ", "")
    found = [w for w in expect_words if w in s]
    return {"file": path.split("/")[-1], "text_elements": n_text,
            "named_font": named, "opaque_white_rect": white_rect,
            "words_found": f"{len(found)}/{len(expect_words)}",
            "kb": round(len(s) / 1024, 1)}
