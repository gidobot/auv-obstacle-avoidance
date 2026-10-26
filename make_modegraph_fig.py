"""Mode transition graph for the no-livelock argument.

The relation was enumerated exhaustively from the deployed selection
(cpp/src/occupancy_map.cpp: set_mode_from_cmd_depth, the Step 2c latch
override and the Step 3 tail hook) over previous mode, vertical error, latch
state and tail state, and over 56 threshold/hysteresis pairs: 24 of the 25
ordered pairs occur, the branch taken when both zero-surge predicates hold is
dead, and the single absent edge is OBSTACLE_CLEAR -> ALT_CORRECTION.

Visual structure follows docs/traversing-the-cliff-manifold.html: only the two
states the proof is about are boxed, the zero-progress region is an outline
rather than a fill, the terminal state is dashed, and latch-mediated edges
carry their own colour. Laid out for a single column, so the proportions and
type sizes differ from the web original.

Node boxes are sized from the rendered text, so labels cannot overflow them.

    python3 make_modegraph_fig.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = ("/home/gidobot/workspace/acfr/auv-obstacle-avoidance/docs/paper/"
       "6aa364bd4d6d76d5dae0d68b/figures/modegraph")

INK, MUTED, RULE, SURF2 = "#0b0b0b", "#52514e", "#b8c7d0", "#eef2f7"
ABSENT, LATCH = "#eb6834", "#2a78d6"
FS_NODE, FS_SUB, FS_EDGE, FS_REG = 5.3, 4.3, 4.7, 4.5
MONO = {"family": "monospace"}
SANS = {"family": "sans-serif"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.monospace": ["DejaVu Sans Mono"], "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavuserif", "font.size": 7,
})

fig, ax = plt.subplots(figsize=(3.45, 2.3))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7.4)
ax.axis("off")

boxes = {}
PADX, PADY = 0.30, 0.0


def width_of(txt, fs, font):
    t = ax.text(0, -99, txt, fontsize=fs, **font)
    fig.canvas.draw()
    w = t.get_window_extent().transformed(ax.transData.inverted()).width
    t.remove()
    return w


def node(cx, name, sub, y0, y1, dashed=False):
    """Box sized from the identifier; the gloss sits below it, unboxed."""
    half = 0.5 * width_of(name, FS_NODE, MONO)
    x0, x1 = cx - half - PADX, cx + half + PADX
    ax.add_patch(FancyBboxPatch((x0, y0), x1 - x0, y1 - y0,
                                boxstyle="round,pad=0,rounding_size=0.10",
                                facecolor="none" if dashed else "white",
                                edgecolor=INK, lw=0.6,
                                linestyle=(0, (2.2, 1.8)) if dashed else "-",
                                zorder=3))
    ax.text(cx, 0.5 * (y0 + y1), name, fontsize=FS_NODE, color=INK,
            ha="center", va="center", zorder=4, **MONO)
    ax.text(cx, y0 - 0.25, sub, fontsize=FS_SUB, color=MUTED, ha="center",
            va="center", zorder=4, **SANS)
    boxes[name] = (x0, x1, y0, y1)


def arrow(p0, p1, color=MUTED, ls="-", lw=0.7):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=5.0,
                                 lw=lw, color=color, linestyle=ls, shrinkA=0,
                                 shrinkB=0, zorder=5))


RX0, RX1 = 0.15, 9.85

# forward-progress region: a region, not a set of objects -- bare labels
ax.add_patch(FancyBboxPatch((RX0, 5.30), RX1 - RX0, 1.65,
                            boxstyle="round,pad=0,rounding_size=0.14",
                            facecolor=SURF2, edgecolor=RULE, lw=0.5, zorder=0))
ax.text(RX0 + 0.20, 6.70, "FORWARD PROGRESS  ·  $v_x = v_s$", fontsize=FS_REG,
        color=MUTED, ha="left", va="center", zorder=1, **MONO)
for cx, nm in ((2.05, "ALT_FOLLOW"), (5.00, "OBSTACLE_HOLD"),
               (7.95, "TAIL_CLEAR")):
    ax.text(cx, 5.85, nm, fontsize=FS_NODE, color=INK, ha="center",
            va="center", zorder=2, **MONO)

# zero-progress region: outlined in the accent, so the eye lands on it
ax.add_patch(FancyBboxPatch((RX0, 0.95), RX1 - RX0, 3.15,
                            boxstyle="round,pad=0,rounding_size=0.14",
                            facecolor="none", edgecolor=ABSENT, lw=0.6,
                            linestyle=(0, (3.0, 2.4)), alpha=0.7, zorder=0))
ax.text(RX0 + 0.20, 1.18, "ZERO PROGRESS  ·  $v_x = 0$", fontsize=FS_REG,
        color=ABSENT, ha="left", va="center", zorder=1, **MONO)

node(2.20, "ALT_CORRECTION", "descends · target bounded", 2.95, 3.85)
node(5.60, "OBSTACLE_CLEAR", "ascends · target monotone", 2.95, 3.85)
node(8.60, "SURFACE HOLD", "target unreachable", 2.95, 3.85, dashed=True)

ac, oc, sh = (boxes["ALT_CORRECTION"], boxes["OBSTACLE_CLEAR"],
              boxes["SURFACE HOLD"])
mid = 0.5 * (ac[1] + oc[0])

# the single edge of the induced subgraph, and the one that is absent
arrow((ac[1], 3.55), (oc[0], 3.55), lw=0.8)
ax.text(mid, 3.90, r"$\Delta z \leq -T$", fontsize=FS_EDGE, color=MUTED,
        ha="center", va="center", **SANS)
arrow((oc[0], 3.15), (ac[1], 3.15), color=ABSENT, ls=(0, (2.4, 1.6)), lw=0.8)
for dx in (-1, 1):
    ax.plot([mid - 0.10 * dx, mid + 0.10 * dx], [3.05, 3.25], lw=1.0,
            color=ABSENT, zorder=6, solid_capstyle="round")
ax.text(mid, 2.28, "removed by the\nascent guard", fontsize=FS_EDGE,
        color=ABSENT, ha="center", va="center", linespacing=1.2, **SANS)

# entries and exits; label side and row chosen so none crosses an arrow
for x, y, lbl, col, up, ha in (
        (1.70, 4.95, r"$\Delta z \geq +T$", MUTED, False, "right"),
        (2.80, 4.65, "reached",             MUTED, True,  "left"),
        (4.55, 4.95, r"$\Delta z \leq -T$", MUTED, False, "right"),
        (5.90, 4.65, "reached",             MUTED, True,  "right"),
        (6.50, 4.35, "latch holds",         LATCH, True,  "left")):
    top = ac[3] if x < 4.0 else oc[3]
    lw = 0.85 if col == LATCH else 0.7
    arrow((x, top), (x, 5.30), col, lw=lw) if up else \
        arrow((x, 5.30), (x, top), col, lw=lw)
    ax.text(x + (0.12 if ha == "left" else -0.12), y, lbl, fontsize=FS_EDGE,
            color=col, ha=ha, va="center", **SANS)

# the sole terminus, reached only when the target cannot be attained
arrow((oc[1], 3.40), (sh[0], 3.40), ls=(0, (2.0, 1.6)))

# legend: three edge classes
for x, col, ls, lbl in ((0.30, MUTED, "-", "transition"),
                        (2.65, LATCH, "-", "latch-mediated"),
                        (5.55, ABSENT, (0, (2.4, 1.6)), "absent by construction")):
    arrow((x, 0.42), (x + 0.45, 0.42), col, ls=ls)
    ax.text(x + 0.57, 0.42, lbl, fontsize=FS_EDGE, color=MUTED, ha="left",
            va="center", **SANS)

fig.savefig(OUT + ".pdf", bbox_inches="tight", pad_inches=0.01,
            metadata={"CreationDate": None})
fig.savefig("/tmp/modegraph_preview.png", dpi=300, bbox_inches="tight",
            pad_inches=0.01)
print("wrote", OUT + ".pdf")
