"""Schematic of the commit-release manoeuvre for the Method section.

Conceptual, not data: it names the geometry that Sec. IV-D states in symbols
(the latch window, the tracked peak, the release position, the interval over
which the crest is unobservable) so the two recorded panels of the
representation figure can be read against it.

    python3 make_manoeuvre_fig.py
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch

OUT = ("/home/gidobot/workspace/acfr/auv-obstacle-avoidance/docs/paper/"
       "6aa364bd4d6d76d5dae0d68b/figures/manoeuvre")

C_MANIFOLD, C_COMMAND = "#2a78d6", "#eb6834"
INK, INK2, GRID_INK, BASE_INK = "#0b0b0b", "#52514e", "#c9c8c3", "#9a9892"
ASTAR, L, S = 2.0, 2.0, 2.0

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 7,
    "axes.labelsize": 7.5, "axes.linewidth": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
})

# an unfollowable face: the gradient between -3.2 and -0.6 exceeds v_z/v_x
TX = np.array([-15, -8, -4.5, -3.2, -2.8, -2.2, -1.4, -0.6, 0,
               1.0, 2.2, 3.6, 5.5, 8.0, 10.0])
TZ = np.array([12.2, 12.0, 11.8, 11.5, 9.5, 7.2, 5.7, 5.05, 5.0,
               5.3, 6.2, 7.6, 9.4, 10.8, 11.2])
terr = lambda q: np.interp(q, TX, TZ)

X_PEAK, Z_PEAK = 0.0, 5.0
X_COMMIT = -3.2
X_REL = X_PEAK + S + L / 2                 # release is a distance, not a time
Z_DAGGER = Z_PEAK - ASTAR

fig, ax = plt.subplots(figsize=(3.45, 2.05))

# terrain
xf = np.linspace(-15, 10, 600)
ax.fill_between(xf, terr(xf), 14.0, color="#d8d7d2", lw=0, zorder=1)
ax.plot(xf, terr(xf), color=C_MANIFOLD, lw=1.3, zorder=3)

# the latched interval: the decision outlives the evidence across this span
ax.add_patch(Rectangle((X_COMMIT, 1.6), X_REL - X_COMMIT, 12.4 - 1.6,
                       facecolor="#f3eae4", edgecolor="none", zorder=0))
for xv in (X_COMMIT, X_REL):
    ax.plot([xv, xv], [1.6, 12.4], color=C_COMMAND, lw=0.6,
            ls=(0, (2.4, 1.6)), zorder=2)

# flown trajectory: follow, climb in place, hold ahead, resume
seg_x = np.linspace(-15, X_COMMIT, 60)
ax.plot(seg_x, terr(seg_x) - ASTAR, color=C_COMMAND, lw=1.5, zorder=5)
ax.plot([X_COMMIT, X_COMMIT], [terr(X_COMMIT) - ASTAR, Z_DAGGER],
        color=C_COMMAND, lw=1.5, zorder=5)
ax.plot([X_COMMIT, X_REL], [Z_DAGGER, Z_DAGGER], color=C_COMMAND, lw=1.5,
        zorder=5)
# after release the descent is rate limited, then rejoins altitude following
tail_x = np.linspace(X_REL, 10, 120)
ax.plot(tail_x, np.minimum(Z_DAGGER + (tail_x - X_REL), terr(tail_x) - ASTAR),
        color=C_COMMAND, lw=1.5, zorder=5)

# the latched target, drawn as the level the command holds
ax.plot([X_COMMIT - 1.4, X_REL + 1.2], [Z_DAGGER, Z_DAGGER], color=BASE_INK,
        lw=0.7, ls=(0, (1.1, 1.4)), zorder=4)
ax.text(X_REL + 1.4, Z_DAGGER, r"$z^{\dagger}$", color=INK2, fontsize=7,
        va="center", ha="left")

# vehicle, at commit and holding over the crest
for xv, zv in ((X_COMMIT, terr(X_COMMIT) - ASTAR), (X_PEAK, Z_DAGGER)):
    ax.add_patch(Rectangle((xv - L / 2, zv - 0.33), L, 0.66, facecolor="white",
                           edgecolor=INK, lw=0.7, zorder=7))

# s + L/2: why release is where it is
ax.annotate("", xy=(X_PEAK, 2.5), xytext=(X_REL, 2.5),
            arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK2,
                            shrinkA=0, shrinkB=0), zorder=8)
ax.text(0.5 * (X_PEAK + X_REL), 2.1, r"$s + L/2$", fontsize=6.4, color=INK2,
        ha="center", va="bottom")

# imaging altitude, over followable ground
ax.annotate("", xy=(-11.0, terr(-11.0)), xytext=(-11.0, terr(-11.0) - ASTAR),
            arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK2,
                            shrinkA=0, shrinkB=0), zorder=8)
ax.text(-10.6, terr(-11.0) - ASTAR / 2, r"$a^{\star}$", fontsize=7,
        color=INK2, va="center")

ax.plot([X_PEAK], [Z_PEAK], marker="o", ms=2.8, mfc=C_MANIFOLD, mec="white",
        mew=0.5, zorder=6)
ax.text(X_PEAK + 1.05, 4.25, r"$x_{peak}$", fontsize=6.4, color=INK2,
        ha="left", va="center")
ax.text(-1.75, 4.25, "latched", fontsize=6.4, color=C_COMMAND, ha="center",
        va="center")
ax.text(X_COMMIT - 0.35, 6.2, "climb in place,\nzero surge", fontsize=6.2,
        color=INK2, ha="right", va="center", linespacing=1.15)

ax.text(X_COMMIT - 0.3, 2.35, "commit", fontsize=6.6, color=C_COMMAND,
        ha="right", va="center")
ax.text(X_REL + 0.3, 2.35, "release", fontsize=6.6, color=C_COMMAND,
        ha="left", va="center")


ax.set_xlim(-15, 10)
ax.set_ylim(12.9, 1.6)
ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("along-track position (m)")
ax.set_ylabel("depth (m)")
for s_ in ("top", "right"):
    ax.spines[s_].set_visible(False)
for s_ in ("left", "bottom"):
    ax.spines[s_].set_color(INK2)
ax.tick_params(colors=INK2, width=0.5, length=2.5)
ax.set_yticks([2, 5, 8, 11])
fig.tight_layout(pad=0.25)
fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.01,
            metadata={"CreationDate": None})
fig.savefig("/tmp/manoeuvre_preview.png", dpi=240, bbox_inches="tight",
            pad_inches=0.01)
print("commit %.1f  peak %.1f  release %.1f  z_dagger %.1f -> %s.pdf"
      % (X_COMMIT, X_PEAK, X_REL, Z_DAGGER, OUT))
