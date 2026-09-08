"""Three-way controller comparison for the publication baseline.

Runs the identical mission — same terrain, same trajectory, same sensor models,
same perception stack — through three controllers and reports the metrics the
paper needs:

    latch       the deployed supervisory commitment controller (C++ core)
    tp          set-based task priority, memoryless
    tp+dwell    set-based task priority with activation dwell, i.e. commitment
                reintroduced inside the task-priority formalism

The hypothesis under test is that `tp` collides or loses clearance on terrain
where the obstacle stops being observable mid-maneuver, that `latch` does not,
and that `tp+dwell` recovers most of the difference — which is the argument that
the commitment mechanism is necessary rather than stylistic.

Usage:
    python compare_controllers.py                 # all terrains
    python compare_controllers.py --terrain step  # one
    python compare_controllers.py --csv out.csv
"""

import argparse
import csv
import math
import sys

import numpy as np

import occupancy_map_cpp as cpp
from simulator import Simulator, make_terrain
from task_priority_baseline import TaskPriorityMapper


# ── controllers under test ───────────────────────────────────────────────────

def latch_factory(cfg, dvl, sonar, alt):
    return cpp.ObstacleMapper(cfg, dvl, sonar, alt)

def tp_factory(cfg, dvl, sonar, alt):
    return TaskPriorityMapper(cfg, dvl, sonar, alt, dwell_s=0.0)

def tp_dwell_factory(cfg, dvl, sonar, alt):
    # Dwell sized to the committed traverse the geometry demands:
    # (cliff_standoff + vehicle_length) / survey_speed.
    hold = (cfg.cliff_standoff + cfg.vehicle_length) / max(cfg.survey_speed, 1e-6)
    return TaskPriorityMapper(cfg, dvl, sonar, alt, dwell_s=hold)

CONTROLLERS = {
    'latch':    latch_factory,
    'tp':       tp_factory,
    'tp+dwell': tp_dwell_factory,
}


# ── metrics ──────────────────────────────────────────────────────────────────

def run(controller, terrain_fn, steps, dt, initial_depth, cfg):
    sim = Simulator(
        omap_config=cfg,
        terrain_fn=terrain_fn,
        initial_depth=initial_depth,
        debug=False,
        mapper_factory=CONTROLLERS[controller],
    )
    clearance, alt_err, modes = [], [], []
    collided = False
    first_collision_x = math.nan
    half = cfg.vehicle_length / 2.0
    # Sample across the whole hull, not just the origin.  A tail collision is
    # precisely the case where the nadir reading looks fine while the stern is
    # in the terrain, so an origin-only metric cannot see the failure under test.
    offsets = np.linspace(-half, half, 9)

    for _ in range(steps):
        sim.step(dt)
        hull = np.array([terrain_fn(sim.vehicle_x + o) for o in offsets], dtype=float)
        clr = float(np.min(hull - sim.vehicle_z))     # worst point on the hull
        nadir = terrain_fn(sim.vehicle_x) - sim.vehicle_z

        clearance.append(clr)
        if np.isfinite(nadir):
            alt_err.append(nadir - cfg.imaging_altitude)
        if clr <= 0.0 and not collided:
            collided = True
            first_collision_x = sim.vehicle_x
        mode = getattr(sim.mapper, 'control_mode', None) or sim.mapper.omap.control_mode
        modes.append(mode)

    transitions = sum(1 for a, b in zip(modes, modes[1:]) if a != b)
    clearance = np.array(clearance, dtype=float)
    return {
        'controller':    controller,
        'min_clearance': float(np.nanmin(clearance)),
        'mean_clearance': float(np.nanmean(clearance)),
        'alt_rms':       float(np.sqrt(np.mean(np.square(alt_err)))) if alt_err else math.nan,
        'collided':      collided,
        'collision_x':   first_collision_x,
        'transitions':   transitions,
        'distance_m':    float(sim.vehicle_x),
    }


# ── terrains ─────────────────────────────────────────────────────────────────

def build_terrains():
    """Scenarios from the simulator's own terrain library.

    `sawtooth` is the occlusion case, and it is the reason the library already
    has it: each tooth rises to a crest and then drops away vertically.  The
    moment the crest passes behind the nose the forward window sees open water,
    while the stern is still over the edge — descend now and the tail catches.
    Because the pattern repeats, every run aggregates many crossings rather than
    resting on a single event.

    `sawtooth-rev` mirrors it: a vertical wall to climb, then a descending
    slope.  The hazard is on approach rather than departure, so a reactive
    controller has the obstacle in view throughout and should do comparatively
    well — included so the comparison shows where task priority is adequate.

    `default` is the library's undulating seafloor with several cliff features,
    as a mixed, less contrived case.

    Each entry is (terrain_fn, initial_depth).
    """
    return {
        'default':      (make_terrain('default'), 18.0),
        'sawtooth':     (make_terrain('sawtooth', slope_angle_deg=30.0, amplitude=8.0,
                                      base_depth=20.0, flat_bottom=6.0), 18.0),
        'sawtooth-rev': (make_terrain('sawtooth', slope_angle_deg=30.0, amplitude=8.0,
                                      base_depth=20.0, flat_bottom=6.0, reverse=True), 18.0),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--terrain', choices=sorted(build_terrains()) + ['all'], default='all')
    ap.add_argument('--steps', type=int, default=1400)
    ap.add_argument('--dt', type=float, default=0.1)
    ap.add_argument('--csv', metavar='PATH', help='also write the results as CSV')
    args = ap.parse_args()

    cfg = cpp.OccupancyMapConfig()
    terrains = build_terrains()
    names = sorted(terrains) if args.terrain == 'all' else [args.terrain]

    rows = []
    for name in names:
        terrain_fn, z0 = terrains[name]
        print(f"\n{name}  ({args.steps} steps @ {args.dt}s, imaging altitude "
              f"{cfg.imaging_altitude:g} m)")
        print(f"  {'controller':<10} {'min clr':>8} {'alt rms':>8} "
              f"{'trans':>6} {'collision':>10}")
        print(f"  {'-'*10} {'-'*8:>8} {'-'*8:>8} {'-'*6:>6} {'-'*10:>10}")
        for ctrl in CONTROLLERS:
            r = run(ctrl, terrain_fn, args.steps, args.dt, z0, cfg)
            r['terrain'] = name
            rows.append(r)
            coll = f"x={r['collision_x']:.1f}m" if r['collided'] else "—"
            print(f"  {ctrl:<10} {r['min_clearance']:8.3f} {r['alt_rms']:8.3f} "
                  f"{r['transitions']:6d} {coll:>10}")

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")

    print("\nmin clr = minimum terrain clearance (m); <= 0 is a collision.")
    print("alt rms = RMS deviation from the imaging altitude (m).")
    print("trans   = controller mode / task activation changes.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
