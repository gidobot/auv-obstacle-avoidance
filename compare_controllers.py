"""Controller comparison for the publication baseline.

Runs the identical mission — same terrain, same trajectory, same sensor models,
same perception stack — through three controllers and reports the metrics the
paper needs:

    latch       the deployed supervisory commitment controller (C++ core)
    tp          set-based task priority, no controller state
    tp+dwell    set-based task priority with activation dwell, i.e. commitment
                reintroduced inside the task-priority formalism

Objective
---------
The survey exists to image the seafloor from a fixed altitude, so the figure of
merit is time held at that altitude with safe standoff — not ground covered.
Stopping to ascend or descend onto altitude before moving on is correct, and
the metrics below score it that way.

Mission
-------
A 4-leg lawnmower flown across a 70-degree sawtooth ridge field: 20 m teeth on a
30 m seafloor with 10 m flats between them, so the vehicle repeatedly climbs
from 28 m to 8 m and back.  Legs run along X, the axis the teeth progress along,
so every 40 m leg crosses roughly two and a third crests.  The turn at the end
of each leg reverses heading, far exceeding the stale-heading gate, which wipes
the occupancy map — that is the condition under which controller state, rather
than map memory, is the only thing that can carry a commitment.

Sweeps
------
`--sweep horizon` varies the map's backward horizon.  The occupancy map retains
terrain behind the vehicle, so a task-priority controller whose safety task
scans the tail band is not memoryless in the way a purely reactive controller
would be.  Shortening the backward horizon removes that map memory and isolates
the contribution of controller state.

`--sweep speed` varies survey speed against a fixed vertical speed.  When the
vehicle climbs as fast as it advances the tail-clearance geometry is never
tight; the hazard should appear as forward speed outruns depth rate.

Usage:
    python compare_controllers.py
    python compare_controllers.py --sweep horizon
    python compare_controllers.py --sweep speed
    python compare_controllers.py --sweep all --csv results.csv
"""

import argparse
import csv
import math
import sys

import numpy as np

import occupancy_map_cpp as cpp
from simulator import (
    Simulator3D,
    make_terrain_3d_sawtooth,
    make_lawnmower_trajectory,
)
from task_priority_baseline import TaskPriorityMapper


# -- mission definition ------------------------------------------------------

SAWTOOTH = dict(
    slope_angle_deg=70.0,   # near-vertical gradual edge
    amplitude=20.0,         # peak-to-trough height
    base_depth=30.0,        # seafloor at the trough
    flat_bottom=10.0,       # flat run between teeth
    orientation_deg=0.0,    # teeth progress along +X
)

LAWNMOWER = dict(
    leg_length=40.0,
    spacing=2.0,
    n_legs=4,
    orientation_deg=0.0,    # legs along X - across the teeth
)

# The vehicle launches at the surface and descends onto the terrain, as it does
# on a real deployment.  The descent is excluded from the metrics below: it says
# nothing about obstacle avoidance and its clearance and altitude error would
# otherwise dominate every statistic.
START_DEPTH = 0.0


def build_mission(survey_speed):
    terrain = make_terrain_3d_sawtooth(**SAWTOOTH)
    # turn_rate=0 gives square corners rather than carved arcs, so each leg
    # transition is two 90-degree heading changes with a straight cross-track
    # run between them.  Simulator3D executes those as on-the-spot yaws: the
    # vehicle holds station, turns, runs the spacing, turns again.  Arc corners
    # would instead keep surging through the turn and would silently widen the
    # pattern whenever the radius exceeded half the leg spacing.
    traj, path_xy = make_lawnmower_trajectory(
        survey_speed=survey_speed, turn_rate=0.0, **LAWNMOWER)
    return terrain, traj, path_xy


# -- controllers under test --------------------------------------------------

def latch_factory(cfg, dvl, sonar, alt):
    return cpp.ObstacleMapper(cfg, dvl, sonar, alt)

def tp_factory(cfg, dvl, sonar, alt):
    return TaskPriorityMapper(cfg, dvl, sonar, alt, dwell_s=0.0)

def tp_dwell_factory(cfg, dvl, sonar, alt):
    # Dwell sized to the committed traverse the geometry demands:
    # (cliff_standoff + vehicle_length) / survey_speed.
    hold = (cfg.cliff_standoff + cfg.vehicle_length) / max(cfg.survey_speed, 1e-6)
    return TaskPriorityMapper(cfg, dvl, sonar, alt, dwell_s=hold)

def tp_stop_factory(cfg, dvl, sonar, alt):
    # Temporal sequencing bolted onto the arbitration: hold station until the
    # altitude task has converged, using the same overshoot band the deployed
    # controller uses for its own mode selection.
    return TaskPriorityMapper(cfg, dvl, sonar, alt, dwell_s=0.0,
                              stop_to_converge=True)

def tp_both_factory(cfg, dvl, sonar, alt):
    # Both additions at once: commitment (dwell on set-based tasks) and
    # sequencing (hold station until the altitude task converges).  This is the
    # strongest form of the objection that the supervisory mode structure is
    # unnecessary — if task priority plus these two rules matches the deployed
    # controller, the mode structure is one way of writing something the
    # formalism can also express, and the paper must say so.
    hold = (cfg.cliff_standoff + cfg.vehicle_length) / max(cfg.survey_speed, 1e-6)
    return TaskPriorityMapper(cfg, dvl, sonar, alt, dwell_s=hold,
                              stop_to_converge=True)

CONTROLLERS = {
    'latch':         latch_factory,
    'tp':            tp_factory,
    'tp+dwell':      tp_dwell_factory,
    'tp+stop':       tp_stop_factory,
    'tp+dwell+stop': tp_both_factory,
}


# -- run + metrics -----------------------------------------------------------

def run(controller, cfg, dt=0.1, margin_s=120.0, alt_tol=0.5, seed=0):
    """Fly the whole lawnmower; return time-on-altitude and safety stats.

    The survey objective is to spend as much of the mission as possible at the
    imaging altitude while keeping safe standoff — not to cover ground quickly.
    Halting to ascend or descend onto altitude before moving on is the correct
    behaviour, so pattern coverage is not a figure of merit and is not reported
    as one.  What matters is time held on altitude, and the along-track distance
    flown while on altitude, which is the usable survey line the mission exists
    to acquire.
    """
    # The sonar model adds unseeded Gaussian range noise, so runs are not
    # reproducible unless the global RNG is pinned.  Seeding per trial — with
    # the same seed across controllers — makes the comparison like-for-like:
    # every controller sees the identical noise realisation.
    np.random.seed(seed)
    terrain, traj, _ = build_mission(cfg.survey_speed)
    sim = Simulator3D(
        omap_config=cfg,
        terrain_fn=terrain,
        trajectory=traj,
        initial_depth=START_DEPTH,
        debug=False,
        mapper_factory=CONTROLLERS[controller],
    )

    path_len = (LAWNMOWER['leg_length'] * LAWNMOWER['n_legs']
                + LAWNMOWER['spacing'] * (LAWNMOWER['n_legs'] - 1))
    # Path time, plus the descent from the surface, plus margin for the in-place
    # ascents during which the vehicle holds station.
    descent_s = SAWTOOTH['base_depth'] / max(cfg.vertical_speed, 1e-6)
    steps = int((path_len / max(cfg.survey_speed, 1e-6) + descent_s + margin_s) / dt)

    half = cfg.vehicle_length / 2.0
    offsets = np.linspace(-half, half, 9)

    clearance, alt_err, modes = [], [], []
    collided, first_collision = False, math.nan
    on_survey = False
    descent_steps = 0
    on_alt_cycles = 0
    dist_on_alt = 0.0
    prev_arc = 0.0

    for _ in range(steps):
        sim.step(dt)
        h = getattr(sim, 'vehicle_heading', 0.0)
        ch, sh = math.cos(h), math.sin(h)
        # Sample the whole hull along the heading axis.  A tail strike is exactly
        # the case where the nadir reading looks fine while the stern is in
        # terrain, so an origin-only metric cannot see the failure under test.
        hull = np.array([terrain(sim.vehicle_x + o * ch, sim.vehicle_y + o * sh)
                         for o in offsets], dtype=float)
        clr = float(np.min(hull - sim.vehicle_z))
        nadir = terrain(sim.vehicle_x, sim.vehicle_y) - sim.vehicle_z

        # Metrics start once the vehicle has first descended onto survey
        # altitude; everything before that is the launch transit.
        if not on_survey:
            descent_steps += 1
            if np.isfinite(nadir) and nadir <= cfg.imaging_altitude + 1.0:
                on_survey = True
            continue

        clearance.append(clr)
        d_arc = max(0.0, float(sim.arc_length) - prev_arc)
        prev_arc = float(sim.arc_length)
        if np.isfinite(nadir):
            err = nadir - cfg.imaging_altitude
            alt_err.append(err)
            if abs(err) <= alt_tol:
                on_alt_cycles += 1
                dist_on_alt += d_arc
        if clr <= 0.0 and not collided:
            collided, first_collision = True, float(getattr(sim, 'arc_length', math.nan))
        mode = getattr(sim.mapper, 'control_mode', None) or sim.mapper.omap.control_mode
        modes.append(mode)

    if not clearance:
        return {'controller': controller, 'min_clearance': math.nan,
                'p05_clearance': math.nan, 'alt_rms': math.nan, 'collided': False,
                'collision_s': math.nan, 'breaches': 0, 'transitions': 0,
                'steps': steps, 'descent_s': descent_steps * dt,
                'note': 'never reached survey altitude'}
    clearance = np.asarray(clearance, dtype=float)
    n = len(clearance)
    return {
        'descent_s':     descent_steps * dt,
        'controller':    controller,
        'on_alt_pct':    100.0 * on_alt_cycles / n if n else math.nan,
        'dist_on_alt':   dist_on_alt,
        'survey_s':      n * dt,
        'min_clearance': float(np.nanmin(clearance)),
        'p05_clearance': float(np.nanpercentile(clearance, 5)),
        'alt_rms':       float(np.sqrt(np.mean(np.square(alt_err)))) if alt_err else math.nan,
        'collided':      collided,
        'collision_s':   first_collision,
        'breaches':      int(np.sum(clearance < 0.5)),
        'transitions':   sum(1 for a, b in zip(modes, modes[1:]) if a != b),
        'steps':         steps,
    }


def trials(controller, cfg, seeds):
    """Aggregate several noise realisations.

    Averages the mission-quality figures and takes the worst case on safety —
    a controller that collides on one seed in five has a safety problem, not a
    slightly lower mean.
    """
    rs = [run(controller, cfg, seed=s) for s in seeds]
    return {
        'controller':    controller,
        'n':             len(rs),
        'on_alt_pct':    float(np.mean([r['on_alt_pct'] for r in rs])),
        'on_alt_sd':     float(np.std([r['on_alt_pct'] for r in rs])),
        'dist_on_alt':   float(np.mean([r['dist_on_alt'] for r in rs])),
        'alt_rms':       float(np.mean([r['alt_rms'] for r in rs])),
        'min_clearance': float(np.min([r['min_clearance'] for r in rs])),
        'breaches':      int(np.sum([r['breaches'] for r in rs])),
        'collisions':    int(np.sum([1 for r in rs if r['collided']])),
    }


def header(title, n):
    print(f"\n{title}   [{n} noise seeds]")
    print(f"  {'controller':<14} {'t@alt %':>13} {'d@alt m':>8} {'alt rms':>8} "
          f"{'worst clr':>9} {'<0.5m':>6} {'collided':>9}")
    print(f"  {'-'*14} {'-'*13} {'-'*8} {'-'*8} {'-'*9} {'-'*6} {'-'*9}")

def show(r):
    coll = f"{r['collisions']}/{r['n']}" if r['collisions'] else "-"
    print(f"  {r['controller']:<14} {r['on_alt_pct']:8.1f}+-{r['on_alt_sd']:<4.1f} "
          f"{r['dist_on_alt']:8.1f} {r['alt_rms']:8.3f} {r['min_clearance']:9.3f} "
          f"{r['breaches']:6d} {coll:>9}")


# -- sweeps ------------------------------------------------------------------

def sweep_none(rows, seeds):
    cfg = cpp.OccupancyMapConfig()
    header(f"nominal  (horizon_back {cfg.horizon_back:g} m, survey {cfg.survey_speed:g} m/s)",
           len(seeds))
    for ctrl in CONTROLLERS:
        r = trials(ctrl, cfg, seeds); r['sweep'], r['value'] = 'nominal', 0.0
        rows.append(r); show(r)

def sweep_horizon(rows, seeds):
    """Backward horizon controls how much terrain the map remembers behind."""
    for hb in (15.0, 8.0, 4.0, 2.0):
        cfg = cpp.OccupancyMapConfig(); cfg.horizon_back = hb
        header(f"backward horizon {hb:g} m", len(seeds))
        for ctrl in CONTROLLERS:
            r = trials(ctrl, cfg, seeds); r['sweep'], r['value'] = 'horizon', hb
            rows.append(r); show(r)

def sweep_speed(rows, seeds):
    """Forward speed against a fixed depth rate tightens the tail geometry."""
    for vx in (0.5, 1.0, 1.5, 2.0):
        cfg = cpp.OccupancyMapConfig(); cfg.survey_speed = vx
        header(f"survey speed {vx:g} m/s  (vertical speed {cfg.vertical_speed:g} m/s)",
               len(seeds))
        for ctrl in CONTROLLERS:
            r = trials(ctrl, cfg, seeds); r['sweep'], r['value'] = 'speed', vx
            rows.append(r); show(r)

SWEEPS = {'none': sweep_none, 'horizon': sweep_horizon, 'speed': sweep_speed}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--sweep', choices=sorted(SWEEPS) + ['all'], default='none')
    ap.add_argument('--csv', metavar='PATH')
    ap.add_argument('--seeds', type=int, default=5,
                    help='noise realisations per condition (default 5)')
    args = ap.parse_args()
    seeds = list(range(args.seeds))

    cfg = cpp.OccupancyMapConfig()
    print(f"mission: {LAWNMOWER['n_legs']} legs x {LAWNMOWER['leg_length']:g} m, "
          f"{LAWNMOWER['spacing']:g} m spacing, legs across the teeth")
    print(f"terrain: {SAWTOOTH['slope_angle_deg']:g} deg sawtooth, "
          f"{SAWTOOTH['amplitude']:g} m teeth on a {SAWTOOTH['base_depth']:g} m floor, "
          f"{SAWTOOTH['flat_bottom']:g} m flats "
          f"(crest at {SAWTOOTH['base_depth'] - SAWTOOTH['amplitude']:g} m)")
    print(f"imaging altitude {cfg.imaging_altitude:g} m, "
          f"stale-heading gate {cfg.stale_heading_threshold_deg:g} deg")

    rows = []
    for n in (sorted(SWEEPS) if args.sweep == 'all' else [args.sweep]):
        SWEEPS[n](rows, seeds)

    if args.csv:
        with open(args.csv, 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {args.csv}")

    print("\nt@alt % = share of survey time held within 0.5 m of imaging altitude,")
    print("          mean +- sd across seeds.")
    print("d@alt m = along-track distance flown while on altitude - the usable")
    print("          survey line acquired.  This is the figure of merit.")
    print("alt rms = RMS deviation from the imaging altitude (m).")
    print("min clr = worst clearance over the hull (m); <= 0 is a collision.")
    print("worst clr = worst hull clearance across all seeds; <= 0 is a collision.")
    print("<0.5m   = control cycles inside half a metre of terrain, summed over seeds.")
    print("Pattern coverage is deliberately not scored: halting to reach")
    print("altitude before moving on is correct behaviour for this survey.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
