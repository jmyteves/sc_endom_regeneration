"""Organoid time-course panels shared by Fig. 5 (collagen I) and Fig. 6 (TNFa).

Input is the human-scored tracked set: one row per organoid that both raters
marked INCLUDE, with its equivalent diameter at 24, 48, 72 and 96 h. The unit
of replication is the well (3 wells per condition, 6 vehicle, 3 Matrigel only),
so every test is a mixed model with a well random intercept:

    log(diameter at h) ~ treatment, groups = well

fitted with statsmodels. The ratio exp(coef) and its p-value are what the
kinetics panels print as stars. The same model on the 24 h-relative diameter is
written to the stats table for the record but not drawn.

Panels
  swarm diameter per organoid, 24-96 h, control vs factor, median bar
  kinetics mean diameter per timepoint with the 25-75 % band, stars per timepoint
  wells 2 x 4 montage of the representative well at each timepoint with the
            tracked representative organoid boxed and a 800 um cut-out inset

Entry points (all return a dict {label: path}):
  build_pair(repo_root, out_dir, ctrl, fac, colour, stem_swarm, stem_kinetics, title=None)
  build_wells(repo_root, out_dir, ctrl, fac, colour, stem)

Reads data/raw_light/organoids/per_organoid_humanscored.csv
        data/raw_light/organoids/representative_organoids.csv
        data/raw_light/organoids/tracks/<well>_tracks.csv
        data/raw_light/organoids/images/<well>_<hour>h.png
"""
import os
import warnings

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image

import render_svg

Image.MAX_IMAGE_PIXELS = None

HOURS = [24, 48, 72, 96]
GREY = "#748190"          # control colour (epithelial slate grey of the atlas palette)
META = "#6b6b6b"
COLOUR = {"TNFα": "#c2185b", "Collagen I": "#6a3d9a"}
# Condition names as they appear in the scored table -> short label on the panel
SHORT = {"Vehicle": "Vehicle", "TNFα 10 ng/mL": "TNFα",
         "Matrigel only": "Matrigel only", "10% collagen I": "Collagen I"}
LONG = {v: k for k, v in SHORT.items()}
UM_PER_PX = 2.50          # projection PNGs; tracking ran at 2x downsample (5.00 um/px)
FS = 10                   # font size of the swarm / kinetics panels (3.5 x 1.9 in slot)
W, H = 3.5, 1.9


def _p(repo_root, *parts):
    return os.path.join(repo_root, "data", "raw_light", "organoids", *parts)


def _need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"expected input missing: {path}")
    return path


def load_scored(repo_root):
    """Wide table, one row per consensus-scored organoid, diameters in um per hour."""
    w = pd.read_csv(_need(_p(repo_root, "per_organoid_humanscored.csv")))
    for h in HOURS:
        if f"diam{h}_um" not in w.columns:
            raise ValueError(f"per_organoid_humanscored.csv lacks diam{h}_um")
    w = w.rename(columns={f"diam{h}_um": h for h in HOURS})
    unknown = set(w["condition"]) - set(SHORT)
    if unknown:
        raise ValueError(f"unexpected condition label(s): {sorted(unknown)}")
    return w


def long_table(wide):
    """One row per organoid x hour: diam_um, area_um2, rel (diameter / own 24 h)."""
    rows = []
    for _, r in wide.iterrows():
        for h in HOURS:
            rows.append(dict(well=r["well"], track=r["track"], condition=SHORT[r["condition"]],
                             hour=h, diam_um=r[h], area_um2=np.pi / 4 * r[h] ** 2,
                             rel=r[h] / r[24]))
    return pd.DataFrame(rows)


def mixed_ratio(wide, resp, ctrl, fac):
    """exp(coef) and p for log(resp) ~ treatment with a well random intercept.

    Tries REML, then ML, then alternative optimisers: a few of these small fits
    (3 + 3 wells) fail to converge under the default settings.
    """
    import statsmodels.formula.api as smf
    d = wide[wide["condition"].isin([ctrl, fac])].copy()
    d["y"] = np.log(d[resp].astype(float))
    d["g"] = (d["condition"] == fac).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for kw in (dict(reml=True), dict(reml=False), dict(reml=False, method="lbfgs"),
                   dict(reml=True, method="cg")):
            try:
                r = smf.mixedlm("y ~ g", d, groups=d["well"]).fit(**kw)
            except Exception:
                continue
            if np.isfinite(r.pvalues["g"]):
                return float(np.exp(r.params["g"])), float(r.pvalues["g"])
    return np.nan, np.nan


def pair_stats(wide, ctrl_short, fac_short):
    """Absolute-diameter and 24 h-relative tests at every timepoint for one pair."""
    ctrl, fac = LONG[ctrl_short], LONG[fac_short]
    w = wide.copy()
    for h in HOURS[1:]:
        w[f"rel{h}"] = w[h] / w[24]
    rows = []
    for h in HOURS:
        r, p = mixed_ratio(w, h, ctrl, fac)
        rows.append(dict(comparison=f"{fac_short} vs {ctrl_short}", metric="absolute diameter",
                         hour=h, ratio=r, p=p))
    for h in HOURS[1:]:
        r, p = mixed_ratio(w, f"rel{h}", ctrl, fac)
        rows.append(dict(comparison=f"{fac_short} vs {ctrl_short}", metric="relative to 24 h",
                         hour=h, ratio=r, p=p))
    out = pd.DataFrame(rows)
    out["n_ctrl"] = int((wide["condition"] == ctrl).sum())
    out["n_fac"] = int((wide["condition"] == fac).sum())
    return out


def stars(p):
    if not np.isfinite(p) or p >= 0.05:
        return "n.s."
    if p >= 0.01:
        return "*"
    if p >= 0.001:
        return "**"
    return "***"


def swarm_x(y, center, hw, ybin):
    """Spread points sharing a y bin symmetrically around center, within +/- hw."""
    y = np.asarray(y, float)
    x = np.full(len(y), float(center))
    b = np.floor((y - np.nanmin(y)) / ybin).astype(int)
    for bb in np.unique(b):
        idx = np.where(b == bb)[0]
        k = len(idx)
        off = (np.arange(k) - (k - 1) / 2) / max(k, 1) * 2 * hw if k > 1 else np.array([0.0])
        x[idx] = center + off
    return x


def _axes():
    fig, ax = plt.subplots(figsize=(W, H),
                           gridspec_kw=dict(left=0.20, right=0.99, bottom=0.25, top=0.86))
    ax.tick_params(labelsize=FS, length=2.5, pad=2, width=0.6)
    return fig, ax


def panel_swarm(S, ctrl, fac, col, stem, ctrl_label=None):
    """Diameter of every organoid at each timepoint, control (grey) beside factor."""
    n_c = int(((S["hour"] == 24) & (S["condition"] == ctrl)).sum())
    n_f = int(((S["hour"] == 24) & (S["condition"] == fac)).sum())
    fig, ax = _axes()
    for j, h in enumerate(HOURS):
        for k, c in enumerate([ctrl, fac]):
            v = S[(S["hour"] == h) & (S["condition"] == c)]["diam_um"].values
            xc = j + (-0.21 if k == 0 else 0.21)
            ax.scatter(swarm_x(v, xc, 0.19, 14), v, s=6, color=GREY if k == 0 else col,
                       alpha=0.65, lw=0)
            ax.hlines(np.median(v), xc - 0.2, xc + 0.2, color="black", lw=1.0, zorder=3)
    ax.set_xticks(range(4))
    ax.set_xticklabels([str(h) for h in HOURS])
    ax.set_xlim(-0.55, 3.55)
    ax.set_ylim(0, 780)
    ax.set_yticks([0, 200, 400, 600])
    ax.set_xlabel("Time (h)", fontsize=FS, labelpad=2)
    ax.set_ylabel("Diameter (µm)", fontsize=FS, labelpad=2)
    for c, cc, n in [(ctrl_label or ctrl, GREY, n_c), (fac, col, n_f)]:
        ax.scatter([], [], s=18, color=cc, label=f"{c} (n = {n})")
    ax.legend(loc="lower left", bbox_to_anchor=(-0.22, 1.0), ncol=2, fontsize=FS - 1,
              frameon=False, handletextpad=0.2, borderaxespad=0.0, columnspacing=0.8)
    render_svg.save_fixed(fig, stem, png_dpi=600)
    plt.close(fig)
    return stem + ".svg"


def panel_kinetics(S, T, ctrl, fac, col, stem, title=None):
    """Mean diameter per timepoint, 25-75 % band, mixed-model star per timepoint."""
    fig, ax = _axes()
    hi = 0
    for c, cc in [(ctrl, GREY), (fac, col)]:
        g = S[S["condition"] == c].groupby("hour")["diam_um"]
        mn, q25, q75 = g.mean(), g.quantile(0.25), g.quantile(0.75)
        ax.fill_between(mn.index, q25, q75, color=cc, alpha=0.12 if c == ctrl else 0.18, lw=0)
        ax.plot(mn.index, mn.values, "-o", color=cc, ms=5, lw=1.4, zorder=3, label=c)
        hi = max(hi, q75.max())
    abs_t = T[(T["metric"] == "absolute diameter")].set_index("hour")
    for h in HOURS:
        p = abs_t.loc[h, "p"]
        top = (S[(S["hour"] == h) & (S["condition"].isin([ctrl, fac]))]
               .groupby("condition")["diam_um"].quantile(0.75).max())
        ax.text(h, top + 6, stars(p), ha="center", va="bottom",
                fontsize=FS - 2 if p >= 0.05 else FS + 1)
    ax.set_xticks(HOURS)
    ax.set_xlim(18, 102)
    ax.set_ylim(80, hi + 48)
    ax.set_xlabel("Time (h)", fontsize=FS, labelpad=2)
    ax.set_ylabel("Diameter (µm)", fontsize=FS, labelpad=2)
    ax.legend(loc="upper left", fontsize=FS - 1, frameon=False, handlelength=1.2,
              borderaxespad=0.1, labelspacing=0.25, handletextpad=0.4)
    if title:
        ax.set_title(title, fontsize=FS, loc="left", pad=3)
    render_svg.save_fixed(fig, stem, png_dpi=600)
    plt.close(fig)
    return stem + ".svg"


def build_pair(repo_root, out_dir, ctrl, fac, col, stem_swarm, stem_kinetics, title=None, ctrl_label=None):
    """Swarm + kinetics for one control / factor pair; returns {label: path}."""
    os.makedirs(out_dir, exist_ok=True)
    render_svg.arial_mode(base=FS, secondary=FS - 1, tick=FS)
    wide = load_scored(repo_root)
    S = long_table(wide)
    T = pair_stats(wide, ctrl, fac)
    out = {
        "swarm": panel_swarm(S, ctrl, fac, col, os.path.join(out_dir, stem_swarm), ctrl_label=ctrl_label),
        "kinetics": panel_kinetics(S, T, ctrl, fac, col, os.path.join(out_dir, stem_kinetics), title=title),
    }
    T.to_csv(os.path.join(out_dir, stem_kinetics + "_mixed_model.csv"), index=False)
    out["stats"] = T
    return out


# Well montages

def load_tracks(repo_root, well):
    """Tracking output for one well: x, y (half-resolution px) and eqd_px per hour."""
    return pd.read_csv(_need(_p(repo_root, "tracks", f"{well}_tracks.csv")))


def representative(repo_root, condition):
    """(well, track) chosen for the montage of one condition."""
    rep = pd.read_csv(_need(_p(repo_root, "representative_organoids.csv")))
    r = rep[rep["condition"] == condition]
    if r.empty:
        raise ValueError(f"no representative organoid listed for {condition!r}")
    return r.iloc[0]["well"], int(r.iloc[0]["track"])


def _image(repo_root, well, hour):
    return Image.open(_need(_p(repo_root, "images", f"{well}_{hour}h.png"))).convert("L")


def _normalise(a, lo_q, hi_q):
    lo, hi = np.percentile(a, [lo_q, hi_q])
    return np.clip((a - lo) / (hi - lo + 1e-6), 0, 1)


def build_wells(repo_root, out_dir, ctrl, fac, col, stem, zoom_field=320, scale=2):
    """2 x 4 whole-well montage (rows: control, factor; columns: 24-96 h).

    The tracked representative organoid is boxed; the inset shows a
    zoom_field x zoom_field px cut-out (800 um at 2.50 um/px) with its diameter.
    At 96 h a 100 um bar is drawn in the inset and on the well. The hue of the
    frame, box and leader is the factor colour; the control row uses grey.
    """
    os.makedirs(out_dir, exist_ok=True)
    hue = {ctrl: GREY, fac: col}
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), gridspec_kw=dict(wspace=0.05, hspace=0.15))
    picks = {}
    for r, c in enumerate([ctrl, fac]):
        w, t = representative(repo_root, c)
        picks[c] = (w, t)
        tr = load_tracks(repo_root, w)
        for j, h in enumerate(HOURS):
            ax = axes[r, j]
            q = tr[(tr["track"] == t) & (tr["hour"] == h)]
            if q.empty:
                raise ValueError(f"track {t} of well {w} has no row at {h} h")
            q = q.iloc[0]
            x, y = q["x"] * 2, q["y"] * 2          # tracking coordinates are half resolution
            im = _image(repo_root, w, h)
            small = np.asarray(im.resize((im.width // scale, im.height // scale), Image.LANCZOS),
                               dtype=np.float32) / 255
            ax.imshow(_normalise(small, 1, 99.5), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor(hue[c]); s.set_linewidth(1.3)
            Hh, Ww = small.shape
            bx, by, bw = x / scale, y / scale, zoom_field / scale
            ax.add_patch(mpatches.Rectangle((bx - bw / 2, by - bw / 2), bw, bw, fill=False,
                                            edgecolor=hue[c], lw=1.0))
            # inset goes to the corner diagonally opposite the organoid
            if bx < Ww / 2 and by < Hh / 2:
                loc = "lower right"
            elif bx >= Ww / 2 and by < Hh / 2:
                loc = "lower left"
            elif bx < Ww / 2:
                loc = "upper right"
            else:
                loc = "upper left"
            ins = inset_axes(ax, width="46%", height="46%", loc=loc, borderpad=0.3)
            crop = im.crop((int(x - zoom_field / 2), int(y - zoom_field / 2),
                            int(x + zoom_field / 2), int(y + zoom_field / 2)))
            ins.imshow(_normalise(np.asarray(crop, dtype=np.float32), 0.5, 99.5),
                       cmap="gray", vmin=0, vmax=1)
            ins.set_xticks([]); ins.set_yticks([])
            for s in ins.spines.values():
                s.set_edgecolor(hue[c]); s.set_linewidth(1.0)
            ins.text(6, 14, f"{q['eqd_px'] * UM_PER_PX:.0f} µm", fontsize=5, color="black",
                     va="top", bbox=dict(facecolor="white", edgecolor="none", alpha=0.75, pad=0.6))
            if j == 3:
                bar = 100 / UM_PER_PX                      # 100 um in full-resolution px
                ins.plot([zoom_field - 10 - bar, zoom_field - 10], [zoom_field - 14] * 2,
                         color="black", lw=1.8, solid_capstyle="butt")
                L = 100 / (scale * UM_PER_PX)
                y0 = Hh * 0.94
                x1 = Ww * 0.96 if loc != "lower right" else Ww * 0.52
                ax.plot([x1 - L, x1], [y0, y0], color="black", lw=2.2, solid_capstyle="butt")
            tx = Ww * 0.77 if "right" in loc else Ww * 0.23
            ty = Hh * 0.77 if "lower" in loc else Hh * 0.23
            ax.plot([bx, tx], [by, ty], color=hue[c], lw=0.6, ls=":")
            if r == 0:
                ax.set_title(f"{h} h", fontsize=7, pad=2)
            if j == 0:
                ax.set_ylabel(c, fontsize=7, color=hue[c], fontweight="bold")
    with mpl.rc_context({"savefig.bbox": "tight"}):
        fig.savefig(stem + ".png", dpi=600, transparent=True)
        fig.savefig(stem + ".svg", transparent=True)
    plt.close(fig)
    return {"wells": stem + ".svg", "picks": picks}
