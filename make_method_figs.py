"""Generate the Method-section representation figure (both panels).

Drives the deployed C++ controller (occupancy_map_cpp) against a terrain
profile sampled from the Tofua reconstruction, and snapshots one frame per
mechanism:

  (a) commitment  -- OBSTACLE_HOLD, the command latched flat ahead
  (b) sequencing  -- TAIL_CLEAR, descent withheld while the stern is blocked

The simulated forward sonar adds Gaussian noise from numpy's global RNG
(simulator.py), so SEED is set here; without it no run reproduces and the
selected frames drift. Re-running this file regenerates both PDFs exactly.

    python3 make_method_figs.py
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

sys.path.insert(0, "/home/gidobot/workspace/acfr/auv-obstacle-avoidance")
import terrain_source as ts
from simulator import Simulator

SEED = 0
MESH = "/home/gidobot/workspace/acfr/acfr_sim_ws/terrain/tofua.obj"
OUTDIR = ("/home/gidobot/workspace/acfr/auv-obstacle-avoidance/docs/paper/"
          "6aa364bd4d6d76d5dae0d68b/figures/")

C_MANIFOLD, C_COMMAND = "#2a78d6", "#eb6834"      # validated pair, CVD dE >= 24
C_PAST, C_TRACK, C_BAND = "#f4b093", "#2f2e2c", "#fbe7dd"   # behind / flown / held
INK, INK2, GRID_INK, BASE_INK = "#0b0b0b", "#52514e", "#c9c8c3", "#9a9892"
ASTAR, L, DS, B = 2.0, 2.0, 2.0, 1.0

plt.rcParams.update({
    # Type 42, not matplotlib's default Type 3: IEEE's PDF check
    # rejects Type 3 outright.
    "pdf.fonttype": 42,

    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "mathtext.fontset": "dejavuserif", "font.size": 7,
    "axes.labelsize": 7.5, "axes.linewidth": 0.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 6.5,
})

field = ts.from_mesh(MESH, cell=0.25, north_axis="y", east_axis="x", up_axis="z")


def run(p0, p1, want, score):
    """Fly the leg under SEED; return a snapshot of the best-scoring frame."""
    np.random.seed(SEED)
    tf = field.along(p0, p1)
    sim = Simulator(terrain_fn=tf, initial_depth=max(0.0, tf(0.0) - ASTAR),
                    debug=False)
    best = None
    for i in range(3200):
        sim.step(0.1)
        om = sim.mapper.omap
        if om.control_mode == want:
            snap = dict(
                step=i, mode=om.control_mode, vz=sim.vehicle_z,
                alt=float(sim.mapper.get_altitude()),
                grid=np.array(om.grid), mz=np.array(om.manifold_z),
                obs=np.array(om.manifold_observed, dtype=bool),
                cd=np.array(om.cmd_depth), nx=om.nx, nz=om.nz, cx=om.cx,
                dx=om.cfg.dx, dz=om.cfg.dz,
                ox=om.grid_origin_x, oz=om.grid_origin_z,
                vx=om.grid_to_world_x(om.cx))
            s = score(snap)
            if s is not None and (best is None or s > best[0]):
                best = (s, snap)
        if sim.vehicle_x >= tf.length - 0.5:
            break
    if best is None:
        raise SystemExit("no %s frame found on this leg" % want)
    return best[1]


def base_of(d):
    return np.where(d["obs"], d["mz"] - ASTAR, np.nan)


def score_commit(d):
    cx, nx = d["cx"], d["nx"]
    if not d["obs"][cx]:
        return None
    ao = np.arange(cx, nx)[d["obs"][cx:]]
    if len(ao) < 10:
        return None
    rise = d["mz"][cx] - d["mz"][ao].min()
    lift = float(np.nanmax(base_of(d)[cx:]) - d["cd"][cx])
    return rise + 0.5 * min(lift, 4.0)


def fly_record(p0, p1):
    """Pass one: mode, along-track position and depth at every step."""
    np.random.seed(SEED)
    tf = field.along(p0, p1)
    sim = Simulator(terrain_fn=tf, initial_depth=max(0.0, tf(0.0) - ASTAR),
                    debug=False)
    rec = []
    for _ in range(3200):
        sim.step(0.1)
        rec.append((sim.mapper.omap.control_mode, sim.vehicle_x, sim.vehicle_z))
        if sim.vehicle_x >= tf.length - 0.5:
            break
    return rec


def snapshots_at(p0, p1, steps):
    """Pass two: re-fly under the same seed and keep the named frames.

    Cheaper than carrying a grid per step: the run is deterministic, so the
    second pass reproduces the first exactly.
    """
    np.random.seed(SEED)
    tf = field.along(p0, p1)
    sim = Simulator(terrain_fn=tf, initial_depth=max(0.0, tf(0.0) - ASTAR),
                    debug=False)
    want, out = set(steps), {}
    for i in range(3200):
        sim.step(0.1)
        if i in want:
            om = sim.mapper.omap
            out[i] = dict(
                step=i, mode=om.control_mode, vz=sim.vehicle_z,
                alt=float(sim.mapper.get_altitude()),
                grid=np.array(om.grid), mz=np.array(om.manifold_z),
                obs=np.array(om.manifold_observed, dtype=bool),
                cd=np.array(om.cmd_depth), nx=om.nx, nz=om.nz, cx=om.cx,
                dx=om.cfg.dx, dz=om.cfg.dz,
                ox=om.grid_origin_x, oz=om.grid_origin_z,
                vx=om.grid_to_world_x(om.cx))
        if len(out) == len(want) or sim.vehicle_x >= tf.length - 0.5:
            if len(out) == len(want):
                break
    return out


def episodes_of(rec, mode):
    """Contiguous runs of one mode, as (first, last) step indices."""
    eps, start = [], None
    for k, r in enumerate(rec):
        if r[0] == mode and start is None:
            start = k
        elif r[0] != mode and start is not None:
            eps.append((start, k - 1))
            start = None
    if start is not None:
        eps.append((start, len(rec) - 1))
    return eps


def pick_tail_episode(rec):
    """The episode that best shows the mechanism.

    Scored on the descent taken once the cap releases and on the relief of
    the ground being crossed, not on how long the cap held: a long hold over
    gentle ground is the common case but makes an illegible picture, because
    nothing visible is being withheld.
    """
    best = None
    for a, b in episodes_of(rec, "TAIL_CLEAR"):
        held = rec[b][1] - rec[a][1]
        after = min(b + 40, len(rec) - 1)
        drop = rec[after][2] - rec[b][2]
        relief = rec[min(b + 160, len(rec) - 1)][2] - rec[a][2]
        if held < 1.0 or drop < 0.5:
            continue
        s = drop * relief
        if best is None or s > best[0]:
            best = (s, a, b)
    if best is None:
        raise SystemExit("no tail-clearance episode with a visible release")
    return best[1], best[2]


def draw(d, out, kind, track=None, held=None, withheld=None):
    nx, nz, cx = d["nx"], d["nz"], d["cx"]
    dx, dz = d["dx"], d["dz"]
    g, mz, obs, cd, vz = d["grid"], d["mz"], d["obs"], d["cd"], d["vz"]
    x = d["ox"] + np.arange(nx) * dx
    z = d["oz"] + np.arange(nz) * dz
    xr = x - d["vx"]
    base = base_of(d)

    fig, ax = plt.subplots(figsize=(3.45, 2.15))
    XE = np.append(xr - dx / 2, xr[-1] + dx / 2)
    ZE = np.append(z - dz / 2, z[-1] + dz / 2)
    # Only occupied cells are drawn. Swept-free cells were shaded too, but
    # the distinction only earns its ink where a column is unobserved, and
    # neither selected frame has one.
    ax.pcolormesh(XE, ZE, np.ma.masked_where(~(g > 0.62), g), cmap="Greys",
                  vmin=0.55, vmax=1.05, shading="flat", rasterized=True,
                  zorder=2)

    if kind == "commit":
        ax.plot(xr, base, color=BASE_INK, lw=1.0, ls=(0, (1.1, 1.4)), zorder=3)
    # observed manifold solid; forward-extended segments dotted
    mo = np.where(obs, mz, np.nan)
    me = np.where(~obs, mz, np.nan)
    ax.plot(xr, mo, color=C_MANIFOLD, lw=1.4, zorder=4)
    ax.plot(xr, me, color=C_MANIFOLD, lw=1.1, ls=(0, (1.0, 1.3)), zorder=4)
    # The command behind the vehicle is what was already flown; ahead of it
    # is the lookahead the controller is still free to revise. Splitting them
    # is what makes the latched segment in (a) legible as a plan rather than
    # a record.
    ax.plot(xr[:cx + 1], cd[:cx + 1], color=C_PAST, lw=1.1,
            ls=(0, (3.2, 1.6)), zorder=5)
    ax.plot(xr[cx:], cd[cx:], color=C_COMMAND, lw=1.5, ls=(0, (3.2, 1.6)),
            zorder=6)

    zlo = min(np.nanmin(mz[obs]), vz) - 2.5
    zhi = max(np.nanmax(mz[obs]), vz) + 2.5

    if kind == "tail":
        ax.axvspan(held[0], held[1], color=C_BAND, zorder=0)
        ax.plot(track[:, 0], track[:, 1], color=C_TRACK, lw=0.9, zorder=7)
        inside = (track[:, 0] >= held[0]) & (track[:, 0] <= held[1])
        ax.plot(track[inside, 0], track[inside, 1], color=C_COMMAND, lw=2.0,
                solid_capstyle="butt", zorder=8)
        # the band is a full-height stripe, so it can be labelled at the top
        # of the axes without a leader crossing the data
        ax.text(0.5 * (held[0] + held[1]), zlo + 0.7,
                "descent withheld\nover %.2f m" % (held[1] - held[0]),
                fontsize=6.2, color=INK2, ha="center", va="top",
                linespacing=1.05, zorder=9)
        ax.annotate("released", xy=(held[1], track[inside, 1][-1]),
                    xytext=(held[1] + 0.6, track[inside, 1][-1] - 1.9),
                    fontsize=6.2, color=INK2, ha="left", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=INK2,
                                    shrinkA=1, shrinkB=2))
    else:
        # the latch's effect shrinks to zero once the baseline catches up, so
        # anchor the arrow at the widest gap rather than a fixed offset
        gap = np.where(np.isfinite(base), np.abs(base - cd), -np.inf)
        gap[:cx] = -np.inf
        j = int(np.argmax(gap))
        ax.annotate("", xy=(xr[j], base[j]), xytext=(xr[j], cd[j]),
                    arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK2,
                                    shrinkA=0, shrinkB=0), zorder=8)
        ax.text(xr[j] + 0.5, 0.5 * (base[j] + cd[j]), "ascent\nheld",
                fontsize=6.2, color=INK2, va="center", linespacing=1.05)

    ax.add_patch(Rectangle((-L / 2, vz - 0.33), L, 0.66, facecolor="white",
                           edgecolor=INK, lw=0.7, zorder=8))
    ax.plot([0], [vz], marker="o", ms=2.6, color=INK, zorder=9)
    ax.axvline(0.0, color=GRID_INK, lw=0.5, ls=":", zorder=1)

    ax.set_xlim(xr[0], xr[-1])
    ax.set_ylim(zhi, zlo)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("along-track position relative to vehicle (m)")
    ax.set_ylabel("depth (m)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, width=0.5, length=2.5)
    ax.grid(True, color=GRID_INK, lw=0.35, alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    handles = [Line2D([], [], color=C_MANIFOLD, lw=1.4)]
    labels = [r"manifold $m(x)$"]
    if (~obs).any():          # only key a style the panel actually draws
        handles.append(Line2D([], [], color=C_MANIFOLD, lw=1.1,
                              ls=(0, (1.0, 1.3))))
        labels.append("extended")
    handles += [Line2D([], [], color=C_COMMAND, lw=1.5, ls=(0, (3.2, 1.6))),
                Line2D([], [], color=C_PAST, lw=1.1, ls=(0, (3.2, 1.6)))]
    labels += [r"command ahead $c(x)$", "command flown"]
    if kind == "commit":
        handles.append(Line2D([], [], color=BASE_INK, lw=1.0,
                              ls=(0, (1.1, 1.4))))
        labels.append(r"baseline $m(x)-a^\star$")
    else:
        handles.append(Line2D([], [], color=C_TRACK, lw=0.9))
        labels.append("flown track")
    # (b) fills its upper band with the withheld-descent label, so its key
    # drops to the clear wedge on the right
    loc = "upper left" if kind == "commit" else "center right"
    leg = ax.legend(handles, labels, loc=loc, frameon=False,
                    handlelength=1.5, borderaxespad=0.2, labelspacing=0.2,
                    handletextpad=0.5, borderpad=0.1)
    for t in leg.get_texts():
        t.set_color(INK2)

    fig.tight_layout(pad=0.25)
    # drop the timestamp so the PDF is byte-reproducible, not merely
    # data-reproducible: a figure that changes bytes every build muddies
    # version control and any claim that the run is repeatable.
    fig.savefig(OUTDIR + out + ".pdf", bbox_inches="tight", pad_inches=0.01,
                metadata={"CreationDate": None})
    fig.savefig("/tmp/%s_preview.png" % out, dpi=220, bbox_inches="tight",
                pad_inches=0.01)
    plt.close(fig)
    print("  %-10s step %4d  %-15s depth %6.2f  alt %5.2f  -> %s.pdf"
          % (kind, d["step"], d["mode"], vz, d["alt"], out))


print("seed %d" % SEED)
a = run((0.0, -45.0), (0.0, 60.0), "OBSTACLE_HOLD", score_commit)   # upslope
draw(a, "manifold", "commit")

DOWN = ((0.0, 60.0), (0.0, -45.0))                                  # downslope
rec = fly_record(*DOWN)
ev0, ev1 = pick_tail_episode(rec)
# show the vehicle past the event: the first settled frame after release, so
# the grid, manifold and command all belong to that one later instant and the
# episode itself is read off the flown track behind it
disp = next((k for k in range(ev1 + 30, min(ev1 + 70, len(rec)))
             if rec[k][0] != "TAIL_CLEAR"), min(ev1 + 40, len(rec) - 1))
snaps = snapshots_at(*DOWN, [ev0, disp])
b, b_ev = snaps[disp], snaps[ev0]
track = np.array([(r[1] - b["vx"], r[2]) for r in rec[:disp + 1]])
held = (rec[ev0][1] - b["vx"], rec[ev1][1] - b["vx"])
withheld = float(np.nanmax(base_of(b_ev)[b_ev["cx"]:]) - b_ev["vz"])
draw(b, "tailclear", "tail", track=track, held=held, withheld=withheld)

print("  (b) episode steps %d-%d, held %.2f m of travel at %.2f m depth;"
      % (ev0, ev1, rec[ev1][1] - rec[ev0][1], rec[ev0][2]))
print("      baseline asked %.2f m deeper at engagement; shown at step %d, "
      "%.1f s later, depth %.2f" % (withheld, disp, (disp - ev1) * 0.1, b["vz"]))
print("  (a) latched lift %.2f m"
      % float(np.nanmax(base_of(a)[a["cx"]:]) - a["cd"][a["cx"]]))
