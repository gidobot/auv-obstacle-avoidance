"""Measure the controller against the terrain-defined ceiling.

Two questions, in order:

  1. How much in-band line does the controller achieve, against the analytic
     ideal for the same terrain?  That gap is the paper's headline number.

  2. How much of the gap is perception rather than control?  Running the same
     controller against a densely sensed map isolates it.  If the oracle run
     closes the gap, the policy is optimal given knowledge and the loss is
     sensing; if it does not, the policy characterisation is wrong.

The oracle replaces perception, not the controller.  A dense fan cast in the
vehicle's vertical plane writes occupancy through the ordinary ray interface, so
the planner sees the representation it always sees — just filled in from truth.
The real three-beam configuration still supplies the altitude estimate, because
`dvl_altitude` is a minimum over beams of range * cos(angle) and a wide fan
collapses it: measured, a 70-degree fan reports altitude a median 13 m
shallower than truth against 2.5 m for the real beams.  An oracle built by
widening the DVL alone is therefore a broken sensor rather than a better one,
and the first version of this script was exactly that.

Runs terminate on transect completion rather than on a step budget: in-band
line is a per-transect quantity and a step-limited run would measure a partial
pattern against a whole-pattern ceiling.

Two measurement details matter enough to state.  The simulator integrates
kinematics from the velocity cached by the *previous* control tick, so distance
covered during a step belongs to the mode computed one step earlier; attributing
it to the mode read afterwards puts several metres into modes that command zero
speed.

And the transect is measured from where survey altitude is first reached, not
from the launch point.  The descent from the surface is a deployment transient
whose length is set by the start depth and the vehicle's vertical speed, not by
the terrain, and the analytic ideal has no model for it — it assumes a vehicle
already at altitude.  Counting those metres against a ceiling that never
budgeted for them charges the controller for the launch.  They are therefore
removed from *both* sides: the ideal is recomputed over the same shortened first
leg, so numerator and denominator span the same ground.  The excluded distance
is reported so the exclusion stays visible.

Usage:
    python evaluate_gap.py                 # nominal terrain, sensed vs oracle
    python evaluate_gap.py --sweep angle   # gap across face steepness
    python evaluate_gap.py --seeds 5
"""

import argparse
import math
import sys

import numpy as np

import occupancy_map_cpp as cpp
from ideal_trajectory import analytic_ideal
from simulator import Simulator3D, make_terrain_3d_sawtooth, make_lawnmower_trajectory
from compare_controllers import SAWTOOTH, LAWNMOWER, START_DEPTH, latch_factory


# ── oracle perception ────────────────────────────────────────────────────────

def _fan(half_fan_deg=70.0, n=48):
    """Beam directions spanning the vehicle's vertical plane, forward and aft."""
    fwd = [(float(a), 0.0) for a in np.linspace(2.0, half_fan_deg, n // 2)]
    aft = [(float(a), 180.0) for a in np.linspace(2.0, half_fan_deg, n // 2)]
    d = cpp.DVLConfig()
    d.beams = fwd + aft
    return d


class OracleMapper:
    """Deployed controller, given a manifold populated from the true terrain.

    Perception is replaced, not the controller.  A dense fan in the vehicle's
    vertical plane is cast against the terrain each DVL cycle and written into
    the occupancy grid through the ordinary ray interface, so the planner sees
    the same representation it always does — just filled in.

    The real three-beam configuration still supplies the altitude estimate.
    That matters: `dvl_altitude` is a minimum of range * cos(angle) across
    hitting beams, so a wide fan collapses it — beams far off nadir contribute
    small vertical components and the minimum follows them down.  Measured, a
    70-degree fan reports altitude a median 13 m shallower than truth against
    2.5 m for the real configuration, which would make the oracle a broken
    sensor rather than a better one.  So the fan writes occupancy and the
    altitude is restored from the real beams afterwards.
    """

    def __init__(self, cfg, dvl_cfg, sonar_cfg, alt_cfg, terrain_fn):
        self._inner = cpp.ObstacleMapper(cfg, dvl_cfg, sonar_cfg, alt_cfg)
        self.cfg = cfg
        self.terrain = terrain_fn
        self._fan = _fan()
        self._dirs = np.asarray(self._fan.beam_directions_3d)
        self._angles = np.asarray(self._fan.beam_angles_rad)
        self._clear = np.ones(len(self._angles), dtype=bool)

    # -- pass-through surface ------------------------------------------------
    @property
    def omap(self):            return self._inner.omap
    def reset(self, pose):     self._inner.reset(pose)
    def update_pose(self, p):  self._inner.update_pose(p)
    def get_altitude(self):    return self._inner.get_altitude()
    def get_control(self):     return self._inner.get_control()
    def set_altimeter_altitude(self, v): self._inner.set_altimeter_altitude(v)

    def update_sensor(self, stype, meas, pose):
        self._inner.update_sensor(stype, meas, pose)
        if stype != cpp.SensorType.DVL:
            return
        alt = self._inner.omap.dvl_altitude          # from the real beams
        omap = self._inner.omap
        world_x = omap.grid_to_world_x(omap.cx)
        ch, sh = math.cos(pose.heading), math.sin(pose.heading)

        n = len(self._angles)
        rng = np.zeros(n); hit = np.zeros(n, dtype=bool)
        for i, (fwd, stbd, down) in enumerate(self._dirs):
            r = 0.1
            while r < self._fan.max_range:
                hx = pose.north + (fwd*ch - stbd*sh) * r
                hy = pose.east  + (fwd*sh + stbd*ch) * r
                if pose.depth + down * r >= self.terrain(hx, hy):
                    rng[i], hit[i] = r, True
                    break
                r += 0.15
            if not hit[i]:
                rng[i] = self._fan.max_range
        omap.update_dvl_ray(rng, self._angles, pose.depth, world_x,
                            hit_surface=hit, vehicle_heading=pose.heading,
                            can_clear=self._clear)
        omap.dvl_altitude = alt                      # undo the fan's corruption
        omap.update(pose.depth, pose.heading)        # replan on the filled map


def oracle_factory(terrain_fn):
    def factory(cfg, dvl, sonar, alt):
        return OracleMapper(cfg, dvl, sonar, alt, terrain_fn)
    return factory


# ── ideal over a lawnmower pattern ───────────────────────────────────────────

def pattern_ideal(terrain3d, cfg, skip=0.0):
    """Analytic in-band ceiling for the whole lawnmower, leg by leg.

    Legs alternate direction, and a sawtooth traversed backwards is not the same
    profile — a gradual rise becomes a vertical drop — so each leg is evaluated
    along its own direction of travel.  Cross-track legs run perpendicular to
    the teeth, where the profile is constant and therefore wholly followable.

    `skip` drops the leading metres of the first leg, so the ceiling spans the
    same ground as a run measured from where survey altitude is first reached
    rather than from the launch point.
    """
    leg, n_legs = LAWNMOWER["leg_length"], LAWNMOWER["n_legs"]
    spacing = LAWNMOWER["spacing"]
    total_in_band, total_len, lost, per_leg = 0.0, 0.0, {}, []

    for i in range(n_legs):
        y = i * spacing
        forward = (i % 2 == 0)
        if forward:
            prof = lambda s, y=y: terrain3d(s, y)
        else:
            prof = lambda s, y=y, leg=leg: terrain3d(leg - s, y)
        r = analytic_ideal(prof, skip if i == 0 else 0.0, leg, cfg)
        per_leg.append(r.intervals)
        total_in_band += r.in_band
        total_len += r.length
        for k, v in r.lost.items():
            lost[k] = lost.get(k, 0.0) + v

    # cross-track runs: terrain constant along them, so followable throughout
    cross = spacing * (n_legs - 1)
    total_in_band += cross
    total_len += cross
    return total_in_band, total_len, lost, per_leg


# ── run to transect completion ───────────────────────────────────────────────

def leg_position(arc, leg_len, spacing):
    """Map pattern arc length to (leg index, distance along leg).

    Returns (None, None) on a cross-track segment.  Derived from arc rather
    than from coordinates so it does not have to re-derive the trajectory's
    turn geometry.
    """
    period = leg_len + spacing
    k = int(arc // period)
    within = arc - k * period
    return (k, within) if within <= leg_len else (None, None)


def run(terrain3d, cfg, factory, seed=0, dt=0.1, alt_tol=0.5, max_s=4000.0,
        per_leg=None):
    """Fly the pattern to completion; return in-band line and its accounting.

    Distance is attributed to the mode that *produced* it — the one computed on
    the previous control tick — because the simulator moves on a cached velocity
    command.  The descent from the surface is returned separately rather than
    bucketed, because the surveyed transect begins where survey altitude is
    first reached.
    """
    np.random.seed(seed)
    traj, _ = make_lawnmower_trajectory(survey_speed=cfg.survey_speed,
                                        turn_rate=0.0, **LAWNMOWER)
    sim = Simulator3D(omap_config=cfg, terrain_fn=terrain3d, trajectory=traj,
                      initial_depth=START_DEPTH,
                      debug=False, mapper_factory=factory)
    path_len = (LAWNMOWER["leg_length"] * LAWNMOWER["n_legs"]
                + LAWNMOWER["spacing"] * (LAWNMOWER["n_legs"] - 1))

    half = cfg.vehicle_length / 2.0
    offs = np.linspace(-half, half, 9)
    in_band, prev_arc, worst, descent = 0.0, 0.0, 9e9, 0.0
    cls = {}
    on_survey = False
    oob = {}
    prev_mode = sim.mapper.omap.control_mode      # the mode that moves step one
    steps = int(max_s / dt)

    for _ in range(steps):
        sim.step(dt)
        d_arc = max(0.0, float(sim.arc_length) - prev_arc)
        prev_arc = float(sim.arc_length)
        nadir = terrain3d(sim.vehicle_x, sim.vehicle_y) - sim.vehicle_z

        if not on_survey and np.isfinite(nadir) and nadir <= cfg.imaging_altitude + 1.0:
            on_survey = True

        if not on_survey:
            descent += d_arc
        elif abs(nadir - cfg.imaging_altitude) <= alt_tol:
            in_band += d_arc
        else:
            oob[prev_mode] = oob.get(prev_mode, 0.0) + d_arc

        # Positional test: was this metre one the terrain *required* to be flown
        # out of band?  Compares the vehicle's along-leg position against the
        # ideal's own compulsory intervals, instead of assuming a mapping from
        # control modes onto the ideal's causes.
        if on_survey and per_leg is not None:
            k, s_leg = leg_position(float(sim.arc_length), LAWNMOWER["leg_length"],
                                    LAWNMOWER["spacing"])
            band_ok = abs(nadir - cfg.imaging_altitude) <= alt_tol
            if k is None or k >= len(per_leg):
                key = "cross-track"
            else:
                compulsory = any(a <= s_leg <= b for a, b, _ in per_leg[k])
                if band_ok:
                    key = "in band (over compulsory)" if compulsory else "in band"
                else:
                    key = "out of band, compulsory" if compulsory else \
                          "out of band, AVOIDABLE"
                    if not compulsory:
                        mk = "   avoidable in " + prev_mode
                        cls[mk] = cls.get(mk, 0.0) + d_arc
            cls[key] = cls.get(key, 0.0) + d_arc

        prev_mode = sim.mapper.omap.control_mode

        h = sim.vehicle_heading
        hull = [terrain3d(sim.vehicle_x + o*math.cos(h), sim.vehicle_y + o*math.sin(h))
                for o in offs]
        if on_survey:
            worst = min(worst, float(np.min(np.asarray(hull) - sim.vehicle_z)))
        if sim.arc_length >= path_len - 1e-6:
            break

    return {"in_band": in_band, "arc": float(sim.arc_length), "oob": oob,
            "complete": sim.arc_length >= path_len - 1e-6, "worst": worst,
            "descent": descent, "cls": cls}


def evaluate(cfg, terrain3d, seeds, label):
    """Run both perception variants and report each against its own ceiling.

    The ceiling is recomputed per run over that run's surveyed transect — the
    pattern less its launch descent — so the ratio compares a run against the
    ideal for the ground it actually covered.  The descent is terrain-
    independent, so the per-run ceilings differ only marginally; the spread is
    reported rather than assumed away.
    """
    # Intervals are taken at skip=0: the launch descent is already excluded from
    # the classifier by the on_survey gate, and clipping leg 0 does not alter
    # any interval lying beyond the clip.
    per_leg = pattern_ideal(terrain3d, cfg)[3]
    rows, ideals, lengths, skips = [], [], [], []
    for name, fac in (("sensed", latch_factory), ("oracle", oracle_factory(terrain3d))):
        rs = [run(terrain3d, cfg, fac, seed=s, per_leg=per_leg) for s in seeds]
        per = [pattern_ideal(terrain3d, cfg, skip=r["descent"]) for r in rs]
        pcts = [100*r["in_band"]/i if i else float("nan")
                for r, (i, _, _, _) in zip(rs, per)]
        ib = float(np.mean([r["in_band"] for r in rs]))
        sd = float(np.std([r["in_band"] for r in rs]))
        done = all(r["complete"] for r in rs)
        worst = float(np.min([r["worst"] for r in rs]))
        oob = {}
        for r in rs:
            for k, v in r["oob"].items():
                oob[k] = oob.get(k, 0.0) + v / len(rs)
        ideals += [i for i, _, _, _ in per]
        lengths += [l for _, l, _, _ in per]
        skips += [r["descent"] for r in rs]
        lost = {}
        for _, _, lo, _ in per:
            for k, v in lo.items():
                lost[k] = lost.get(k, 0.0) + v / len(per)
        cls = {}
        for r in rs:
            for k, v in r["cls"].items():
                cls[k] = cls.get(k, 0.0) + v / len(rs)
        rows.append((name, ib, sd, float(np.mean(pcts)), worst, done, oob,
                     float(np.mean([l for _, l, _, _ in per])), lost, cls))

    length = float(np.mean(lengths))
    ideal  = float(np.mean(ideals))
    skip   = float(np.mean(skips))
    lost = rows[0][8]
    print(f"\n{label}")
    print(f"  surveyed transect {length:.1f} m "
          f"(pattern {length + skip:.1f} m less {skip:.1f} m launch descent)")
    print(f"  analytic ideal {ideal:.1f} m ({100*ideal/length:.1f}% of transect)")
    print(f"    lost to: " + "  ".join(f"{k} {v:.1f} m" for k, v in sorted(lost.items())))
    print(f"  {'run':<8} {'in-band m':>10} {'% of ideal':>11} {'worst clr':>10} {'completed':>10}")
    print(f"  {'-'*8} {'-'*10} {'-'*11} {'-'*10} {'-'*10}")
    for name, ib, sd, pct, worst, done, _, _, _, _ in rows:
        print(f"  {name:<8} {ib:6.1f}±{sd:<3.1f} {pct:10.1f}% {worst:10.3f} {str(done):>10}")
    for name, ib, _, _, _, _, oob, ln, _, cls in rows:
        total = ib + sum(oob.values())
        print(f"\n  {name}: where the {ln:.1f} m went (accounted {total:.1f} m)")
        print(f"     {'in band':<22} {ib:7.1f} m")
        for k, v in sorted(oob.items(), key=lambda kv: -kv[1]):
            print(f"     {k:<22} {v:7.1f} m")
        print(f"     -- against the ideal's own compulsory intervals --")
        for k in ("in band", "in band (over compulsory)", "out of band, compulsory",
                  "out of band, AVOIDABLE", "cross-track"):
            if k in cls:
                print(f"     {k:<28} {cls[k]:7.1f} m")
            if k == "out of band, AVOIDABLE":
                for mk, mv in sorted((x for x in cls.items()
                                      if x[0].startswith("   avoidable")),
                                     key=lambda kv: -kv[1]):
                    print(f"     {mk:<28} {mv:7.1f} m")
    return ideal, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--sweep", choices=["none", "angle"], default="none")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    cfg = cpp.OccupancyMapConfig()

    print(f"followable gradient {cfg.vertical_speed/cfg.survey_speed:g} "
          f"({math.degrees(math.atan(cfg.vertical_speed/cfg.survey_speed)):.0f}°), "
          f"standoff {cfg.safety_standoff_m:g} m, band ±0.5 m, {len(seeds)} seed(s)")

    if args.sweep == "none":
        evaluate(cfg, make_terrain_3d_sawtooth(**SAWTOOTH), seeds,
                 f"{SAWTOOTH['slope_angle_deg']:g}° sawtooth, "
                 f"{SAWTOOTH['amplitude']:g} m teeth")
    else:
        for ang in (20.0, 35.0, 45.0, 55.0, 70.0):
            terr = dict(SAWTOOTH); terr["slope_angle_deg"] = ang
            evaluate(cfg, make_terrain_3d_sawtooth(**terr), seeds,
                     f"{ang:g}° faces  (followable: {'yes' if ang <= 45 else 'no'})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
