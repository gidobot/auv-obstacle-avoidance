"""Behavioural tests for the C++ occupancy map (occupancy_map_cpp).

These replaced a Python-vs-C++ parity suite.  That suite asserted only that the
two implementations *agreed* — never that either was right — so when the Python
prototype was retired it would have left no coverage at all.  The values frozen
here were captured while both implementations were still present and agreeing
exactly (cmd_depth differed by 0.0 across the cliff run), which is what licenses
treating them as a baseline.

Two kinds of assertion, deliberately distinguished:

  * ANALYTIC — ground truth derived from the scenario, independent of the
    implementation.  A regression here is a real bug.
  * BASELINE — the behaviour as of the C++ being validated on the vehicle.
    A change here is not necessarily wrong, but must be deliberate.  Update the
    constant and say why in the commit message.
"""
import math
import numpy as np
import occupancy_map_cpp as cpp

# ── scenario helpers ─────────────────────────────────────────────────────────

def flat_terrain(x):   return 20.0
def cliff_terrain(x):  return 20.0 if x < 10.0 else 5.0

def make_cfg():
    c = cpp.OccupancyMapConfig()
    c.imaging_altitude = 2.0
    c.cliff_standoff   = 2.0
    c.safety_below_m   = 1.0
    c.stale_heading_threshold_deg = 45.0
    return c

def make_mapper():
    m = cpp.ObstacleMapper(make_cfg(), cpp.DVLConfig(), cpp.SonarConfig())
    m.reset(cpp.Pose(north=0, east=0, depth=0, heading=0))
    return m

def cast_dvl(dvl, vehicle_x, vehicle_z, terrain_fn, step=0.1):
    """Ray-march each beam against a height field; returns (ranges, hits).

    Note this caster only resolves terrain *below* the vehicle — it reports the
    first point where the ray passes under the surface.  See test_cliff_run.
    """
    angles = dvl.beam_angles_rad
    ranges = np.zeros(len(angles))
    hits   = np.zeros(len(angles), dtype=bool)
    for i, ang in enumerate(angles):
        r = step
        while r < dvl.max_range:
            if vehicle_z + math.cos(ang) * r >= terrain_fn(vehicle_x + math.sin(ang) * r):
                ranges[i] = r
                hits[i] = True
                break
            r += step
        if not hits[i]:
            ranges[i] = dvl.max_range
    return ranges, hits

def drive(mapper, terrain_fn, steps=60, vehicle_z=18.0, dx=0.5):
    dvl = cpp.DVLConfig()
    modes, cmds = [], []
    for s in range(steps):
        x = s * dx
        ranges, hits = cast_dvl(dvl, x, vehicle_z, terrain_fn)
        mapper.update_sensor(cpp.SensorType.DVL,
                             cpp.DVLMeasurement(ranges, hits),
                             cpp.Pose(north=x, east=0, depth=vehicle_z, heading=0))
        modes.append(mapper.omap.control_mode)
        cmds.append(float(mapper.omap.get_commanded_depth_at_vehicle()))
    return modes, cmds

def mode_runs(seq):
    out = []
    for m in seq:
        if out and out[-1][0] == m:
            out[-1][1] += 1
        else:
            out.append([m, 1])
    return [(m, n) for m, n in out]

VALID_MODES = {'ALT_FOLLOW', 'ALT_CORRECTION', 'OBSTACLE_CLEAR',
               'OBSTACLE_HOLD', 'TAIL_CLEAR', 'DEPTH_HOLD'}

# ── test 1: configuration ────────────────────────────────────────────────────

def test_config():
    c = cpp.OccupancyMapConfig()
    assert abs(c.imaging_altitude - 2.0) < 1e-9
    assert abs(c.prior - 0.5) < 1e-9
    assert not hasattr(c, 'dvl_max_range_m'), \
        "dvl_max_range_m was removed: a real sensor reports its own validity"

    # ANALYTIC — beam geometry follows from the configured slant/heading pairs.
    # This also guards the pybind array path: a stride bug there once made every
    # element of these arrays read back as the last value written.
    dvl = cpp.DVLConfig()
    assert list(dvl.beams) == [(20.0, 0.0), (20.0, 120.0), (20.0, 240.0)]
    expected = [math.atan2(math.sin(math.radians(s)) * math.cos(math.radians(h)),
                           math.cos(math.radians(s)))
                for s, h in dvl.beams]
    got = list(dvl.beam_angles_rad)
    assert np.allclose(got, expected), f"beam angles: {got} != {expected}"
    assert len(set(got)) > 1, \
        "all beam angles identical — stride-0 array regression in the bindings"
    # Only the h=0 beam lies in the vehicle X-Z plane, so only it may clear voxels.
    assert list(dvl.beam_can_clear) == [True, False, False]

    assert abs(cpp.SonarConfig().max_range - 12.0) < 1e-9
    assert abs(cpp.AltimeterConfig().max_range - 100.0) < 1e-9

    mapper = make_mapper()
    ctrl = mapper.get_control()
    assert ctrl.vertical_mode in VALID_MODES
    assert abs(ctrl.vx - c.survey_speed) < 1e-9
    print("PASS test_config")

# ── test 2: altitude over flat terrain ───────────────────────────────────────

def test_flat_terrain_altitude():
    mapper = make_mapper()
    vehicle_z = 18.0                      # flat seafloor at 20 m => altitude 2 m
    ranges, hits = cast_dvl(cpp.DVLConfig(), 0.0, vehicle_z, flat_terrain)
    mapper.update_sensor(cpp.SensorType.DVL,
                         cpp.DVLMeasurement(ranges, hits),
                         cpp.Pose(north=0.0, east=0.0, depth=vehicle_z, heading=0.0))
    alt = mapper.get_altitude()
    # ANALYTIC — true altitude is 2.0 m; the caster quantises range to 0.1 m
    # steps, so allow a little over one step of slack.
    assert abs(alt - 2.0) < 0.15, f"altitude {alt:.4f} should be ~2.0 m"
    print(f"  altitude={alt:.4f} (true 2.0)")
    print("PASS test_flat_terrain_altitude")

def test_flat_terrain_holds_imaging_altitude():
    mapper = make_mapper()
    modes, cmds = drive(mapper, flat_terrain, steps=40, vehicle_z=18.0)
    settled = cmds[10:]
    # ANALYTIC — over flat terrain at 20 m with imaging_altitude 2 m, the
    # commanded depth must converge to 18 m and stay there.
    assert all(abs(c - 18.0) < 0.05 for c in settled), \
        f"cmd_depth should hold ~18.0 over flat terrain, got {settled[:6]}"
    assert set(modes[10:]) <= {'ALT_FOLLOW'}, \
        f"flat terrain at target altitude should stay ALT_FOLLOW, saw {set(modes[10:])}"
    print(f"  settled cmd_depth={settled[0]:.3f}, modes={set(modes[10:])}")
    print("PASS test_flat_terrain_holds_imaging_altitude")

# ── test 3: cliff run (frozen baseline) ──────────────────────────────────────

def test_cliff_run():
    """Step cliff at x=10, vehicle held at 18 m.

    Caveat, so the baseline is not over-read: the terrain rises to 5 m while the
    vehicle stays at 18 m, so past the step the vehicle is *below* the surface
    and the height-field caster returns an immediate hit on every beam.  This
    exercises the OBSTACLE_CLEAR path deterministically, but it is not a
    validation of cliff-climbing — a meaningful one needs the forward sonar and
    a caster that resolves terrain above the vehicle.  Treat the numbers here as
    a change detector, not as proof of correct avoidance.
    """
    mapper = make_mapper()
    modes, cmds = drive(mapper, cliff_terrain, steps=60, vehicle_z=18.0)

    # INVARIANTS — these must hold whatever the tuning.
    assert all(m in VALID_MODES for m in modes), f"unknown mode in {set(modes)}"
    assert all(np.isfinite(c) for c in cmds), "cmd_depth produced a non-finite value"
    assert all(c >= 0.0 for c in cmds), "cmd_depth must never command above the surface"
    # The vehicle starts with no observations and must not be told to dive to the
    # grid floor once terrain is seen.
    assert max(cmds[5:]) <= 20.0, f"cmd_depth {max(cmds[5:])} exceeds the seafloor depth"
    # The cliff must be noticed: the run cannot stay in altitude-following.
    assert 'OBSTACLE_CLEAR' in modes, "step cliff did not trigger OBSTACLE_CLEAR"
    assert modes.index('OBSTACLE_CLEAR') >= 15, \
        "OBSTACLE_CLEAR before the cliff is in range — detection fired too early"
    # The conservative vehicle-column rule must never command a dive before the
    # hull footprint has been observed: no ALT_CORRECTION on the opening cycles.
    assert 'ALT_CORRECTION' not in modes[:5], \
        f"dived before observing terrain under the hull: {modes[:5]}"

    # BASELINE — re-captured when the vehicle-column command became the most
    # conservative of the hull-footprint manifold and the latest direct
    # altitude return.  Two changes from the previous baseline, both expected:
    # the run no longer opens with two ALT_CORRECTION cycles (the old rule dove
    # on a manifold default before anything under the hull had been observed),
    # and settled commands sit exactly on the imaging altitude rather than
    # ~0.07 m below it.  A change here is a behaviour change; make it
    # deliberately and say why.
    assert mode_runs(modes) == [('ALT_FOLLOW', 20), ('OBSTACLE_CLEAR', 40)], \
        f"mode sequence changed: {mode_runs(modes)}"
    for idx, want in [(10, 18.0), (19, 17.4095), (20, 16.0), (59, 16.0)]:
        assert abs(cmds[idx] - want) < 1e-3, \
            f"cmd_depth[{idx}]={cmds[idx]:.4f}, baseline {want}"
    print(f"  mode runs: {mode_runs(modes)}")
    print("PASS test_cliff_run")

# ── test 4: grid snapshot ────────────────────────────────────────────────────

def test_grid_snapshot():
    mapper = cpp.ObstacleMapper(cpp.OccupancyMapConfig(), cpp.DVLConfig(),
                                cpp.SonarConfig())
    mapper.reset(cpp.Pose(north=0, east=0, depth=10, heading=0))
    snap = mapper.omap.get_grid_snapshot()
    assert 'grid' in snap and 'cmd_depth' in snap
    assert snap['grid'].shape == (snap['nz'], snap['nx'])
    assert snap['nx'] == mapper.omap.nx
    assert snap['cx'] == mapper.omap.cx
    # An unobserved grid must read as the prior everywhere, not as zeros —
    # another shape the pybind array bug could take.
    assert np.allclose(snap['grid'], cpp.OccupancyMapConfig().prior), \
        "a fresh grid should be uniformly at the prior"
    print(f"  grid shape: {snap['grid'].shape}  nx={snap['nx']}  cx={snap['cx']}")
    print("PASS test_grid_snapshot")

# ── test 5: cmd_depth array access ───────────────────────────────────────────

def test_cmd_depth_access():
    mapper = cpp.ObstacleMapper(cpp.OccupancyMapConfig(), cpp.DVLConfig(),
                                cpp.SonarConfig())
    mapper.reset(cpp.Pose(north=0, east=0, depth=10, heading=0))
    omap = mapper.omap
    cd = omap.cmd_depth
    assert len(cd) == omap.nx
    assert np.all(np.isfinite(np.asarray(cd))), "cmd_depth contains non-finite entries"
    print(f"  cmd_depth[cx]={cd[omap.cx]}  len={len(cd)}")
    print("PASS test_cmd_depth_access")

# ── run all ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_config()
    test_flat_terrain_altitude()
    test_flat_terrain_holds_imaging_altitude()
    test_cliff_run()
    test_grid_snapshot()
    test_cmd_depth_access()
    print("\nAll tests passed.")
