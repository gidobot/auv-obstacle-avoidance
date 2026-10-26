"""Mode transition graph for the no-livelock argument.

The relation drawn here was enumerated exhaustively from the deployed
selection (cpp/src/occupancy_map.cpp: set_mode_from_cmd_depth, the Step 2c
latch override and the Step 3 tail hook) over previous mode, vertical error,
latch state and tail state, and over 56 threshold/hysteresis pairs: 24 of the
25 ordered pairs occur, the branch taken when both zero-surge predicates hold
is dead, and the single absent edge is OBSTACLE_CLEAR -> ALT_CORRECTION.

Node boxes are sized from the rendered text rather than from guessed widths,
so the labels cannot overflow them.

    python3 make_modegraph_fig.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = ("/home/gidobot/workspace/acfr/auv-obstacle-avoidance/docs/paper/"
       "6aa364bd4d6d76d5dae0d68b/figures/modegraph")

C_COMMAND = "#eb6834"
INK, INK2, GRID_INK, BASE_INK = "#0b0b0b", "#52514e", "#c9c8c3", "#9a9892"
BAND_GO, BAND_STOP = "#eef2f7", "#f6ece6"
FS_NODE, FS_EDGE, FS_NOTE = 5.4, 4.9, 4.6

plt.rcParams.update({
    # Type 42, not matplotlib's default Type 3: IEEE's PDF check
    # rejects Type 3 outright.
    "pdf.fonttype": 42,

    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 7,
})

fig, ax = plt.subplots(figsize=(3.45, 2.3))
ax.set_xlim(0, 10)
ax.set_ylim(0.0, 6.9)
ax.axis("off")

PADX, PADY = 0.28, 0.20
boxes = {}


def node(cx, cy, name, key=None):
    """Place the label, measure it, then fit a box around it."""
    t = ax.text(cx, cy, name, fontsize=FS_NODE, color=INK, ha="center",
                va="center", zorder=4)
    fig.canvas.draw()
    bb = t.get_window_extent().transformed(ax.transData.inverted())
    x0, x1 = bb.x0 - PADX, bb.x1 + PADX
    y0, y1 = bb.y0 - PADY, bb.y1 + PADY
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="white", edgecolor=INK, lw=0.6,
                                zorder=3))
    boxes[key or name] = (x0, x1, y0, y1)


def band(y0, y1, fc, label, x0=0.1, x1=9.9):
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.16",
                                facecolor=fc, edgecolor=GRID_INK, lw=0.5,
                                zorder=0))
    ax.text(x0 + 0.16, y1 - 0.16, label, fontsize=FS_EDGE, color=INK2,
            va="top", ha="left", zorder=1)


def arrow(p0, p1, color=INK2, ls="-", lw=0.7, rad=0.0):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5.0,
                                 lw=lw, color=color, linestyle=ls, shrinkA=0,
                                 shrinkB=0,
                                 connectionstyle="arc3,rad=%g" % rad,
                                 zorder=5))


band(5.05, 6.85, BAND_GO, r"forward progress   $v_x = v_s$")
node(1.85, 5.80, "ALT_FOLLOW")
node(5.00, 5.80, "OBSTACLE_HOLD")
node(8.15, 5.80, "TAIL_CLEAR")

band(1.70, 4.15, BAND_STOP, r"zero progress   $v_x = 0$")
node(2.10, 3.05, "ALT_CORRECTION")
node(7.50, 3.05, "OBSTACLE_CLEAR")
ax.text(2.10, 2.32, "bounded target", fontsize=FS_NOTE, color=INK2,
        ha="center", va="center", style="italic", zorder=4)
ax.text(7.50, 2.32, "monotone target", fontsize=FS_NOTE, color=INK2,
        ha="center", va="center", style="italic", zorder=4)

ac, oc = boxes["ALT_CORRECTION"], boxes["OBSTACLE_CLEAR"]
mid = 0.5 * (ac[1] + oc[0])

# the single edge of the induced subgraph, and the one that is absent
arrow((ac[1], 3.22), (oc[0], 3.22), lw=0.8)
ax.text(mid, 3.38, r"$\Delta z \leq -T$", fontsize=FS_EDGE, color=INK2,
        ha="center", va="bottom")
arrow((oc[0], 2.84), (ac[1], 2.84), color=C_COMMAND, ls=(0, (2.0, 1.3)),
      lw=0.8)
ax.plot([mid], [2.84], marker="x", ms=3.8, mew=1.0, color=C_COMMAND, zorder=6)
ax.text(mid, 2.62, "absent", fontsize=FS_EDGE, color=C_COMMAND, ha="center",
        va="top")

# between the bands; labels staggered so the four do not collide
for x, y, lbl, ha, down, top in (
        (1.40, 4.88, r"$\Delta z \geq +T$",  "left",  True,  ac[3]),
        (3.30, 4.45, "target reached",        "left",  False, ac[3]),
        (6.20, 4.88, r"$\Delta z \leq -T$",  "right", True,  oc[3]),
        (8.95, 4.45, "ascent complete",       "right", False, oc[3])):
    arrow((x, 5.05), (x, top)) if down else arrow((x, top), (x, 5.05))
    ax.text(x + (0.12 if ha == "left" else -0.12), y, lbl, fontsize=FS_EDGE,
            color=INK2, ha=ha, va="center")

# the sole terminus for forward progress
node(7.50, 1.02, "surface hold")
arrow((7.50, oc[2]), (7.50, boxes["surface hold"][3]))
ax.text(7.50, 0.42, "target unreachable", fontsize=FS_NOTE, color=INK2,
        ha="center", va="center", style="italic")

fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.01,
            metadata={"CreationDate": None})
fig.savefig("/tmp/modegraph_preview.png", dpi=260, bbox_inches="tight",
            pad_inches=0.01)
print("wrote", OUT + ".pdf")
