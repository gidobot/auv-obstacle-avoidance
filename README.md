# AUV 2D Obstacle Avoidance Simulator

A 2D occupancy-grid based obstacle avoidance system for a seafloor imaging Autonomous Underwater Vehicle (AUV). The system operates in the vehicle-relative X-Z plane (forward/depth) like a side-scrolling game, ignoring lateral (Y) offsets.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Sensor Observations                       │
│   DVL (3 Janus beams + altimeter)    Forward-looking sonar  │
└──────────────┬────────────────────────────────┬──────────────┘
               │                                │
               ▼                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    OccupancyMap                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  2D Voxel Grid (X: 0.5m bins, Z: 0.25m bins)       │    │
│  │  Vehicle-relative side-scroll window                │    │
│  │  Look-behind ◄── Vehicle ──► Look-ahead             │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Cliff Manifold Extraction                          │    │
│  │  Stair-step polyline over occupied voxels           │    │
│  └─────────────────────────────────────────────────────┘    │
│                         │                                    │
│                         ▼                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Commanded Depth Path                               │    │
│  │  - Imaging altitude above manifold                  │    │
│  │  - Cliff standoff (vertical climb before cliff)     │    │
│  │  - Tail clearance (hold until tail clears)          │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
               │
               ▼
         Commanded Depth → Vehicle Depth Controller
```

## C++ Library

The `cpp/` directory contains a C++17 implementation of `OccupancyMap` with a pybind11 Python binding.

### Dependencies

- CMake ≥ 3.14
- Eigen3 ≥ 3.3
- pybind11

On Ubuntu/Debian:

```bash
sudo apt install cmake libeigen3-dev python3-pybind11
```

Or install pybind11 via pip:

```bash
pip install pybind11
```

### Build

A convenience script in the repo root handles configure, build, and install in one step:

```bash
./build.sh            # Release build
./build.sh --debug    # Debug build
./build.sh --clean    # Wipe build dir first, then build
./build.sh --test     # Build then run test_occupancy_map.py
```

The extension is built for whichever interpreter is first on `PATH` — activate
the project venv before building, or point the script at one explicitly:

```bash
source venv/bin/activate && ./build.sh --test
PYTHON=/path/to/python ./build.sh          # or select one directly
```

This matters: an extension module built against a different Python cannot be
imported, and the install step removes any `occupancy_map_cpp*.so` left over
from a build for another interpreter so a stale module can never shadow the
current one.

Or build manually:

```bash
cd cpp
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cmake --install build
```

The install step copies the compiled `occupancy_map_cpp.so` module into the repo root, making it importable alongside the Python implementation.

`test_occupancy_map.py` exercises the extension module directly — run it after
any change to the core.  It separates analytic assertions (ground truth derived
from the scenario) from frozen baselines (the behaviour as of vehicle
validation); a baseline change is not automatically wrong, but must be
deliberate.

### Targets

| Target | Type | Description |
|--------|------|-------------|
| `auv_oavoid` | Static library | Core C++ implementation, linkable into other C++ projects |
| `occupancy_map_cpp` | Python extension | pybind11 bindings — drop-in replacement for `occupancy_map.py` |

### Use the C++ binding from Python

```python
from occupancy_map_cpp import OccupancyMap, OccupancyMapConfig

config = OccupancyMapConfig()
config.dx = 0.5
config.dz = 0.25
# ... same interface as the Python implementation
omap = OccupancyMap(config)
```

### Link against `auv_oavoid` from another CMake project

`auv_oavoid` is the LCM-free, Python-free core.  Add the `cpp/` directory as a
subdirectory and link the target:

```cmake
add_subdirectory(path/to/auv-obstacle-avoidance/cpp)
target_link_libraries(my_target PRIVATE auv_oavoid)
```

The Python extension is skipped automatically when this project is consumed
this way — pybind11 and a matching Python are not needed by a consumer, only
Eigen.  (Force it either way with `-DAUV_OAVOID_BUILD_PYTHON=ON/OFF`.)

There is no installed CMake package to `find_package`, by design: the core is
two files, and consumers pin a version by vendoring this repo as a git
submodule rather than by installing it.

#### Consumers

`acfr-lcm`'s `oa-mapper` node consumes this repo as a submodule at
`src/acfr/oa-mapper/auv-obstacle-avoidance`, pinned to a specific commit.  The
node builds `oa_mapper.cpp` against `auv_oavoid` from that submodule, so the
algorithm running on the vehicle is exactly the revision recorded by the
submodule pointer.

Changes needed by the vehicle must land **here** and the submodule pointer be
bumped there — never edited in the consuming repo.  The C++ core is the single
implementation of the algorithm; `test_occupancy_map.py` is what guards it.

---

## Files

| File | Description |
|------|-------------|
| `cpp/` | The algorithm: `OccupancyMap` / `ObstacleMapper` — voxel grid, sensor updates, manifold extraction, path planning. The design notes live in `cpp/include/occupancy_map.hpp`. |
| `cpp/python/bindings.cpp` | pybind11 bindings exposing that core to Python as `occupancy_map_cpp`. |
| `simulator.py` | Simulation environment with synthetic terrain, DVL/sonar sensor models, and vehicle motion along the planned path. |
| `visualizer.py` | Browser-based real-time visualizer using WebSocket streaming. |
| `lcm_playback_visualizer.py` | Replays an LCM log (or a live feed) through the mapper; mirrors `oa_mapper.cpp`'s sensor gating. |
| `test_occupancy_map.py` | Behavioural tests for the core. |
| `docs/commitment-under-occlusion.html` | Positioning and section outline for a publication on the method. Open it in a browser. |

> A Python implementation of the algorithm (`occupancy_map.py`) was maintained
> alongside the C++ as the original prototype.  It was removed once the C++ was
> validated on the vehicle: keeping two hand-written implementations in step had
> cost real bugs, and the parity suite that was supposed to catch them asserted
> only that the two agreed, never that either was correct.  The simulator and
> visualizers remain in Python and drive the core through the extension module.
> `git log -- occupancy_map.py` still has it if the reference is ever wanted.

## Quick Start

### Run the visualizer

```bash
pip install websockets
python visualizer.py
# Open http://localhost:8080 in a browser
```

### Run headless simulation

```bash
python simulator.py
```

### Use in your own code

Build the extension first (`./build.sh`), then:

```python
from occupancy_map_cpp import OccupancyMap, OccupancyMapConfig
import numpy as np

# Configure.  The binding exposes plain attributes rather than a keyword
# constructor, so set what you need after default-constructing.
config = OccupancyMapConfig()
config.dx               = 0.5    # X bin size (m)
config.dz               = 0.25   # Z bin size (m)
config.horizon_fwd      = 15.0   # Forward look-ahead (m)
config.horizon_back     = 15.0   # Backward look-behind (m)
config.vehicle_length   = 2.0    # Vehicle length for tail clearance (m)
config.imaging_altitude = 2.0    # Target altitude above seafloor (m)
config.cliff_standoff   = 2.0    # Climb trigger distance before cliff (m)

omap = OccupancyMap(config)
omap.reset(vehicle_world_x=0.0, vehicle_depth=18.0)

# Each control cycle:
# 1. Feed sensor data
omap.update_dvl_ray(dvl_ranges, dvl_beam_angles, vehicle_depth, vehicle_x)
omap.update_sonar(sonar_range, sonar_half_angle, vehicle_depth, vehicle_x, hit)

# 2. Advance grid as vehicle moves
omap.advance(forward_distance_this_cycle)

# 3. Run planning pipeline
cmd_depth = omap.update(vehicle_depth)

# 4. Use commanded depth in your depth controller
depth_error = cmd_depth - vehicle_depth
```

## Parameters

### OccupancyMapConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `dx` | 0.5 m | Forward axis bin size |
| `dz` | 0.25 m | Depth axis bin size (finer for altitude control) |
| `horizon_fwd` | 15.0 m | Forward look-ahead distance |
| `horizon_back` | 15.0 m | Backward look-behind distance |
| `z_min` | 0.0 m | Minimum depth in grid |
| `z_max` | 40.0 m | Maximum depth in grid |
| `vehicle_length` | 2.0 m | Vehicle length for tail clearance |
| `imaging_altitude` | 2.0 m | Target altitude above seafloor |
| `survey_speed` | 0.5 m/s | Forward survey speed |
| `cliff_standoff` | 2.0 m | Distance before cliff face to begin climb |
| `occ_thresh` | 0.62 | Probability threshold for occupied voxel |
| `dvl_hit_prob` | 0.14 | DVL occupancy increment on hit |
| `sonar_hit_prob` | 0.04 | Sonar occupancy increment (lower — noisier) |

### Sensor adaptation: who decides a beam is valid

`ObstacleMapper` consumes `(range, hit)` per beam and nothing else.  Deciding
whether a return is real is the **adapter's** job — the code that turns a
specific sensor's messages into those pairs.  Each sensor states validity
differently, so a configured max range is not a general answer:

| Sensor | How a no-return is identified | Needs a configured max range? |
|--------|-------------------------------|-------------------------------|
| DVL (`nucleus_bottomtrack_t`) | `distance_beam_valid[i]`, plus `DISTANCE_SENTINEL = 0.0` | **No** — fully self-describing |
| Altimeter (`nucleus_altimeter_t`) | `DISTANCE_SENTINEL = 0.0`, and `altimeter_quality` | **Not in principle** — see note below |
| Forward sonar (`isa500_t`) | nothing — the message is only `distance` | **Yes** — the only available signal |

A real sensor has already applied its own range limit, so clamping a *valid*
return to a configured maximum invents terrain: a genuine 30 m bottom lock
reported as a hit at 25 m is a fabricated obstacle.  `DVLConfig.max_range`
therefore exists for the **simulator only**, where the ray-caster must generate
no-returns itself; it is not a vehicle parameter.

> **TODO — characterise `altimeter_quality`.**  The altimeter path still infers
> a no-return by comparing the range against a configured maximum
> (`range < max_range - 0.05`), in both `oa_mapper.cpp` and
> `lcm_playback_visualizer.py`.  That is a stand-in for information the message
> already carries: `nucleus_altimeter_t` provides `altimeter_quality` and a
> documented zero sentinel.  Replacing it needs a defensible quality threshold
> taken from logged values, which nobody has established yet — so the heuristic
> is kept, identically, on both sides until then.  Note the current thresholds
> differ per vehicle (`cheryl.cfg` sets `altimeter_max_range = 25`), so genuine
> returns beyond that are presently discarded as no-returns.

Both adapters must agree, or a log replay will not reproduce what the vehicle
did.  `lcm_playback_visualizer.py` mirrors `oa_mapper.cpp` deliberately; pass
`--sonar-max-range` / `--altimeter-max-range` to match the vehicle's
`bot_param` values, since the tool's defaults are this repo's reference config
rather than any particular vehicle's.

## Coordinate Conventions

- **X**: Forward axis (positive ahead of vehicle)
- **Z**: Depth axis (positive downward)
- **Y**: Lateral axis (ignored in 2D projection)

The occupancy grid is vehicle-relative and yaw-agnostic. Sensor measurements are projected from 3D into the X-Z plane by ignoring Y offsets. If two DVL beams project into the same (X, Z) bin from different Y positions, they observe the same cell.

## Cliff Avoidance Behavior

1. **Approach**: Vehicle follows terrain at imaging altitude (2m above seafloor)
2. **Detection**: Forward sonar detects cliff face in occupancy grid
3. **Standoff climb**: At standoff distance (2m) before cliff face, path steps vertically to cliff-top altitude
4. **Hold**: Vehicle maintains cliff-top altitude while traversing the cliff
5. **Tail clearance**: Holds altitude until the full vehicle length has cleared the cliff edge
6. **Descent**: Path steps vertically back down to imaging altitude on the far side
