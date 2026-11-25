"""Fig 4a, module trajectories, one panel per module, broken down by compartment.

Rebuild of the three-programme trajectory row (Homeostasis / Pregnancy / Repair) as a
five-module row, each module scored within each of the 4 compartments so the trajectory
cannot be an artefact of which cells are present at a stage.
"""
import numpy as np, matplotlib.pyplot as plt


def build(TRAJ, order, mcol, mpeak, comps, comp_col, stages, stage_lab, outstem, title,
          subtitle=None, n_ct=None, fs=1.0, lw=1.25, ms=4.0, save=None, pw_scale=1.0,
          rotation=45, legend_below=True):
    NB = len(order)
    # canvas grows with the type, otherwise 2x labels collide
    PW, PH = 1.62*pw_scale, 1.62*pw_scale
    GAP, LEFT = 0.30*fs, 1.22*fs
    # legend below means the right margin only needs to clear the last x tick label
    RIGHT = (0.28*fs if legend_below else 1.55*fs)
    LEG_H = (0.30*fs if legend_below else 0.0)
    TOP, HDR = 0.42*fs, 0.40*fs
    BOT = 0.86*fs * (0.80 if 0 < rotation < 90 else 1.0)
    FIG_W = LEFT + NB*PW + (NB-1)*GAP + RIGHT
    FIG_H = TOP + HDR + PH + BOT + LEG_H
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fx, fy = (lambda v: v/FIG_W), (lambda v: v/FIG_H)
    y0 = BOT + LEG_H
    X = np.arange(len(stages))

    vals = np.concatenate([TRAJ[m].values.ravel() for m in order])
    vals = vals[~np.isnan(vals)]
    lo, hi = vals.min(), vals.max()
    pad = 0.10*(hi-lo)

    for bi, m in enumerate(order):
        x0 = LEFT + bi*(PW+GAP)
        ax = fig.add_axes([fx(x0), fy(y0), fx(PW), fy(PH)])
        T = TRAJ[m]
        for c in comps:
            ax.plot(X, T.loc[c].values, color=comp_col[c], lw=lw, zorder=3,
                    marker="o", ms=ms, mec="white", mew=0.7*fs)
        ax.axhline(0, color="#aaaaaa", lw=0.5, ls=(0, (3, 2)), zorder=1)
        # mark the module's own stage
        pk = stages.index(mpeak[m])
        ax.axvspan(pk-0.42, pk+0.42, color=mcol[m], alpha=0.10, lw=0, zorder=0)
        ax.set_xlim(-0.45, len(stages)-0.55); ax.set_ylim(lo-pad, hi+pad)
        ax.set_xticks(X)
        # Angled ticks must anchor at their right edge, or the label drifts away from its
        # tick as rotation moves off 90 deg. rotation_mode='anchor' pins the anchor point.
        ax.set_xticklabels([stage_lab[s] for s in stages], rotation=rotation, fontsize=6.6*fs,
                           ha='right' if 0 < rotation < 90 else 'center',
                           rotation_mode='anchor' if 0 < rotation < 90 else None)
        for t, s in zip(ax.get_xticklabels(), stages):
            t.set_color(mcol[order[stages.index(s)]] if False else "#555555")
        ax.tick_params(length=2*fs, pad=1.6*fs, labelsize=6.6*fs)
        for sp in ("top", "right"): ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"): ax.spines[sp].set_linewidth(0.6)
        if bi == 0:
            ax.set_ylabel("module score\n(mean z across profiles)", fontsize=7.0*fs, labelpad=6*fs)
        else:
            ax.set_yticklabels([])
        # header: module name + n genes + peak stage
        fig.text(fx(x0+PW/2), fy(y0+PH+0.20*fs), m, ha="center", va="bottom",
                 fontsize=8.0*fs, color=mcol[m], fontweight="bold")
        fig.text(fx(x0+PW/2), fy(y0+PH+0.06*fs),
                 f"{TRAJ[m].attrs.get('n', '')} genes \u00b7 peaks {stage_lab[mpeak[m]]}",
                 ha="center", va="bottom", fontsize=6.4*fs, color="#777777")

    # compartment legend, inside the canvas on the right
    h = [plt.Line2D([], [], color=comp_col[c], lw=lw, marker="o", ms=ms,
                    mec="white", mew=0.7*fs, label=c) for c in comps]
    if legend_below:
        # One horizontal row spanning the plot width, title inline at the left so the
        # four entries stay on a single line rather than wrapping under a stacked title.
        fig.legend(handles=h, loc="lower center", bbox_to_anchor=(0.5, fy(0.04*fs)),
                   frameon=False, fontsize=7.4*fs, ncol=len(comps),
                   handletextpad=0.5, columnspacing=1.6*fs,
                   title="compartment", title_fontsize=7.8*fs)
    else:
        fig.legend(handles=h, loc="center left",
                   bbox_to_anchor=(fx(LEFT+NB*PW+(NB-1)*GAP+0.12*fs), fy(y0+PH*0.55)),
                   frameon=False, fontsize=7.4*fs, labelspacing=0.6, handletextpad=0.6,
                   title="compartment", title_fontsize=7.8*fs)

    fig.text(fx(0.10), fy(FIG_H-0.12*fs), title, ha="left", va="top", fontsize=8.6*fs)
    if subtitle:
        fig.text(fx(0.10), fy(FIG_H-0.28*fs), subtitle, ha="left", va="top",
                 fontsize=6.8*fs, color="#777777")
    if save is not None:
        save(fig, outstem)
    return fig
