"""Smoke-test: run identical missions on Python and C++ ObstacleMapper, compare results."""
import sys, math, numpy as np

sys.path.insert(0, '.')

from occupancy_map import (
    OccupancyMapConfig, DVLConfig, SonarConfig, AltimeterConfig,
    Pose, SensorType, DVLMeasurement, AltimeterMeasurement, SonarMeasurement,
    ObstacleMapper as PyObstacleMapper,
)
import occupancy_map_cpp as cpp

# ── helpers ──────────────────────────────────────────────────────────────────

def make_cfg():
    return OccupancyMapConfig(imaging_altitude=2.0, cliff_standoff=2.0,
                               safety_below_m=1.0, stale_heading_threshold_deg=45.0)

def make_cpp_cfg():
    c = cpp.OccupancyMapConfig()
    c.imaging_altitude = 2.0
    c.cliff_standoff   = 2.0
    c.safety_below_m   = 1.0
    c.stale_heading_threshold_deg = 45.0
    return c

def flat_terrain(x): return 20.0          # flat at 20 m
def cliff_terrain(x): return 20.0 if x < 10.0 else 5.0   # step cliff at x=10

# ── test 1: basic import and construction ────────────────────────────────────

def test_construction():
    c = cpp.OccupancyMapConfig()
    assert abs(c.imaging_altitude - 2.0) < 1e-9
    assert abs(c.prior - 0.5) < 1e-9

    dvl = cpp.DVLConfig()
    angles = dvl.beam_angles_rad
    assert len(angles) == 4
    assert abs(angles[0]) < 1e-9           # altimeter is straight down

    mapper = cpp.ObstacleMapper(c, dvl, cpp.SonarConfig())
    ctrl = mapper.get_control()
    assert ctrl.vertical_mode == 'ALT_FOLLOW'
    assert abs(ctrl.vx - c.survey_speed) < 1e-9
    print("PASS test_construction")

# ── test 2: DVL update and altitude ─────────────────────────────────────────

def _run_dvl_step(mapper, vehicle_x, vehicle_z, terrain_fn, dvl_cfg, heading=0.0):
    """Simulate one DVL observation and update."""
    angles = dvl_cfg.beam_angles_rad
    ranges = np.zeros(len(angles))
    hits   = np.zeros(len(angles), dtype=bool)
    for i, ang in enumerate(angles):
        r = 0.1
        while r < dvl_cfg.max_range:
            hx = vehicle_x + math.sin(ang)*r
            hz = vehicle_z + math.cos(ang)*r
            if hz >= terrain_fn(hx):
                ranges[i] = r
                hits[i]   = True
                break
            r += 0.1
        if not hits[i]:
            ranges[i] = dvl_cfg.max_range
    pose = Pose(north=vehicle_x, east=0.0, depth=vehicle_z, heading=heading)
    meas = DVLMeasurement(ranges=ranges, hit_surface=hits)
    mapper.update_sensor(SensorType.DVL, meas, pose)
    return ranges, hits

def test_dvl_altitude():
    dvl_py  = DVLConfig()
    dvl_cpp = cpp.DVLConfig()

    py_mapper  = PyObstacleMapper(make_cfg(),  dvl_py,  SonarConfig())
    cpp_mapper = cpp.ObstacleMapper(make_cpp_cfg(), dvl_cpp, cpp.SonarConfig())

    py_mapper.reset(Pose(0,0,0,0))
    cpp_mapper.reset(cpp.Pose(north=0,east=0,depth=0,heading=0))

    vehicle_z = 18.0  # 2 m above flat terrain at 20 m
    vehicle_x = 0.0
    _run_dvl_step(py_mapper,  vehicle_x, vehicle_z, flat_terrain, dvl_py)

    # C++ equivalent
    angles = dvl_cpp.beam_angles_rad
    ranges = np.zeros(len(angles))
    hits   = np.zeros(len(angles), dtype=bool)
    for i, ang in enumerate(angles):
        r = 0.1
        while r < dvl_cpp.max_range:
            hx = vehicle_x + math.sin(ang)*r
            hz = vehicle_z + math.cos(ang)*r
            if hz >= flat_terrain(hx):
                ranges[i] = r; hits[i] = True; break
            r += 0.1
        if not hits[i]: ranges[i] = dvl_cpp.max_range

    pose_cpp = cpp.Pose(north=vehicle_x, east=0.0, depth=vehicle_z, heading=0.0)
    meas_cpp = cpp.DVLMeasurement(ranges, hits)
    cpp_mapper.update_sensor(cpp.SensorType.DVL, meas_cpp, pose_cpp)

    py_alt  = py_mapper.get_altitude()
    cpp_alt = cpp_mapper.get_altitude()
    print(f"  Python alt={py_alt:.4f}  C++ alt={cpp_alt:.4f}")
    assert abs(py_alt - cpp_alt) < 0.05, f"altitude mismatch: py={py_alt} cpp={cpp_alt}"
    print("PASS test_dvl_altitude")

# ── test 3: control mode after cliff approach ────────────────────────────────

def test_cliff_modes():
    dvl_py  = DVLConfig()
    dvl_cpp = cpp.DVLConfig()
    py_mapper  = PyObstacleMapper(make_cfg(), dvl_py,  SonarConfig())
    cpp_mapper = cpp.ObstacleMapper(make_cpp_cfg(), dvl_cpp, cpp.SonarConfig())

    py_mapper.reset(Pose(0,0,0,0))
    cpp_mapper.reset(cpp.Pose(north=0,east=0,depth=0,heading=0))

    py_modes, cpp_modes = [], []
    for step in range(60):
        vehicle_x = step * 0.5
        vehicle_z = 18.0

        # Python
        angles = dvl_py.beam_angles_rad
        ranges = np.zeros(len(angles)); hits = np.zeros(len(angles), dtype=bool)
        for i, ang in enumerate(angles):
            r = 0.1
            while r < dvl_py.max_range:
                hx = vehicle_x + math.sin(ang)*r
                hz = vehicle_z + math.cos(ang)*r
                if hz >= cliff_terrain(hx): ranges[i]=r; hits[i]=True; break
                r += 0.1
            if not hits[i]: ranges[i] = dvl_py.max_range
        py_mapper.update_sensor(SensorType.DVL,
                                DVLMeasurement(ranges=ranges, hit_surface=hits),
                                Pose(vehicle_x, 0, vehicle_z, 0))
        py_modes.append(py_mapper.omap.control_mode)

        # C++
        angles_c = dvl_cpp.beam_angles_rad
        ranges_c = np.zeros(len(angles_c)); hits_c = np.zeros(len(angles_c), dtype=bool)
        for i, ang in enumerate(angles_c):
            r = 0.1
            while r < dvl_cpp.max_range:
                hx = vehicle_x + math.sin(ang)*r
                hz = vehicle_z + math.cos(ang)*r
                if hz >= cliff_terrain(hx): ranges_c[i]=r; hits_c[i]=True; break
                r += 0.1
            if not hits_c[i]: ranges_c[i] = dvl_cpp.max_range
        cpp_mapper.update_sensor(cpp.SensorType.DVL,
                                 cpp.DVLMeasurement(ranges_c, hits_c),
                                 cpp.Pose(north=vehicle_x, east=0, depth=vehicle_z, heading=0))
        cpp_modes.append(cpp_mapper.omap.control_mode)

    mismatches = [(i, p, c) for i,(p,c) in enumerate(zip(py_modes, cpp_modes)) if p != c]
    if mismatches:
        print("  Mode mismatches:")
        for s, pm, cm in mismatches[:5]:
            print(f"    step {s}: py={pm} cpp={cm}")
        assert False, f"{len(mismatches)} mode mismatches"
    print(f"  modes: {set(py_modes)}")
    print("PASS test_cliff_modes")

# ── test 4: grid snapshot shape ──────────────────────────────────────────────

def test_grid_snapshot():
    c   = cpp.OccupancyMapConfig()
    dvl = cpp.DVLConfig()
    mapper = cpp.ObstacleMapper(c, dvl, cpp.SonarConfig())
    mapper.reset(cpp.Pose(north=0,east=0,depth=10,heading=0))
    snap = mapper.omap.get_grid_snapshot()
    assert 'grid' in snap and 'cmd_depth' in snap
    assert snap['grid'].shape[1] == snap['nx']
    assert snap['grid'].shape[0] == snap['nz']
    assert snap['nx'] == mapper.omap.nx
    assert snap['cx'] == mapper.omap.cx
    print(f"  grid shape: {snap['grid'].shape}  nx={snap['nx']}  cx={snap['cx']}")
    print("PASS test_grid_snapshot")

# ── test 5: cmd_depth array access ───────────────────────────────────────────

def test_cmd_depth_access():
    c   = cpp.OccupancyMapConfig()
    dvl = cpp.DVLConfig()
    mapper = cpp.ObstacleMapper(c, dvl, cpp.SonarConfig())
    mapper.reset(cpp.Pose(north=0,east=0,depth=10,heading=0))
    omap = mapper.omap
    # cmd_depth should be indexable like a numpy array
    cd = omap.cmd_depth
    assert hasattr(cd, '__len__')
    val = cd[omap.cx]
    print(f"  cmd_depth[cx]={val}")
    print("PASS test_cmd_depth_access")

# ── run all ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_construction()
    test_dvl_altitude()
    test_cliff_modes()
    test_grid_snapshot()
    test_cmd_depth_access()
    print("\nAll tests passed.")
