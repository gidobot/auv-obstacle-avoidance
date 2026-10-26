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
INK, INK2, GRID_INK, BASE_INK = "#0b0b0b", "#52514e", "#c9c8c3", "#9a9892"
ASTAR, L, DS, B = 2.0, 2.0, 2.0, 1.0

plt.rcParams.update({
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


def score_tail(d):
    cx = d["cx"]
    xr = d["ox"] + np.arange(d["nx"]) * d["dx"] - d["vx"]
    sel = ((xr >= -L / 2 - DS) & (xr <= 0) & d["obs"]
           & (d["mz"] >= d["vz"]) & (d["mz"] < d["vz"] + B))
    if sel.sum() == 0:
        return None
    withheld = float(np.nanmax(base_of(d)[cx:]) - d["vz"])
    return 2 * sel.sum() + min(withheld, 6.0)


def draw(d, out, kind):
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
    ax.pcolormesh(XE, ZE, np.ma.masked_where(~(g < 0.5 - 1e-9), g),
                  cmap=matplotlib.colors.ListedColormap(["#e9e8e4"]),
                  shading="flat", rasterized=True, zorder=0)
    ax.pcolormesh(XE, ZE, np.ma.masked_where(~(g > 0.62), g), cmap="Greys",
                  vmin=0.55, vmax=1.05, shading="flat", rasterized=True,
                  zorder=2)

    ax.plot(xr, base, color=BASE_INK, lw=1.0, ls=(0, (1.1, 1.4)), zorder=3)
    # observed manifold solid; forward-extended segments dotted
    mo = np.where(obs, mz, np.nan)
    me = np.where(~obs, mz, np.nan)
    ax.plot(xr, mo, color=C_MANIFOLD, lw=1.4, zorder=4)
    ax.plot(xr, me, color=C_MANIFOLD, lw=1.1, ls=(0, (1.0, 1.3)), zorder=4)
    ax.plot(xr, cd, color=C_COMMAND, lw=1.4, ls=(0, (3.2, 1.6)), zorder=6)

    if kind == "tail":
        ax.add_patch(Rectangle((-L / 2 - DS, vz), DS + L / 2, B,
                               facecolor="none", edgecolor=C_COMMAND, lw=0.7,
                               ls=(0, (2.2, 1.4)), zorder=5))
        sel = ((xr >= -L / 2 - DS) & (xr <= 0) & obs
               & (mz >= vz) & (mz < vz + B))
        ax.plot(xr[sel], mz[sel], ls="none", marker="o", ms=3.0,
                mfc=C_COMMAND, mec="white", mew=0.5, zorder=7)
        j = cx + int(round(6.0 / dx))
        lbl, lx = "descent\nwithheld", xr[j] + 0.5
    else:
        # the latch's effect shrinks to zero once the baseline catches up, so
        # anchor the arrow at the widest gap rather than a fixed offset
        gap = np.where(np.isfinite(base), np.abs(base - cd), -np.inf)
        gap[:cx] = -np.inf
        j = int(np.argmax(gap))
        lbl, lx = "ascent\nheld", xr[j] + 0.5
    ax.annotate("", xy=(xr[j], base[j]), xytext=(xr[j], cd[j]),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=INK2,
                                shrinkA=0, shrinkB=0), zorder=8)
    ax.text(lx, 0.5 * (base[j] + cd[j]), lbl, fontsize=6.2, color=INK2,
            va="center", linespacing=1.05)

    ax.add_patch(Rectangle((-L / 2, vz - 0.33), L, 0.66, facecolor="white",
                           edgecolor=INK, lw=0.7, zorder=8))
    ax.plot([0], [vz], marker="o", ms=2.6, color=INK, zorder=9)
    ax.axvline(0.0, color=GRID_INK, lw=0.5, ls=":", zorder=1)

    zlo = min(np.nanmin(mz[obs]), vz) - 2.5
    zhi = max(np.nanmax(mz[obs]), vz) + 2.5
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
    handles += [Line2D([], [], color=C_COMMAND, lw=1.4, ls=(0, (3.2, 1.6))),
                Line2D([], [], color=BASE_INK, lw=1.0, ls=(0, (1.1, 1.4)))]
    labels += [r"command $c(x)$", r"baseline $m(x)-a^\star$"]
    if kind == "tail":
        handles.append(Line2D([], [], color=C_COMMAND, lw=0.7,
                              ls=(0, (2.2, 1.4))))
        labels.append("tail check")
    loc = "upper left" if kind == "commit" else "upper right"
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
b = run((0.0, 60.0), (0.0, -45.0), "TAIL_CLEAR", score_tail)        # downslope
draw(b, "tailclear", "tail")
xr_b = b["ox"] + np.arange(b["nx"]) * b["dx"] - b["vx"]
selb = ((xr_b >= -L / 2 - DS) & (xr_b <= 0) & b["obs"]
        & (b["mz"] >= b["vz"]) & (b["mz"] < b["vz"] + B))
print("  (b) blocking columns at x_rel %s; withheld %.2f m"
      % (np.round(xr_b[selb], 2),
         float(np.nanmax(base_of(b)[b["cx"]:]) - b["vz"])))
print("  (a) latched lift %.2f m"
      % float(np.nanmax(base_of(a)[a["cx"]:]) - a["cd"][a["cx"]]))
