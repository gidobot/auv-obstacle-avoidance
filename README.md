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
./build.sh --test     # Build then run test_cpp.py
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

`test_cpp.py` runs the same missions through both implementations and asserts
they agree — run it after any change to either backend, since the two are
maintained in parallel and silently diverge otherwise.

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

```cmake
find_package(auv_obstacle_avoidance REQUIRED)
target_link_libraries(my_target PRIVATE auv_oavoid)
```

Or add the `cpp/` directory as a subdirectory:

```cmake
add_subdirectory(path/to/auv-obstacle-avoidance/cpp)
target_link_libraries(my_target PRIVATE auv_oavoid)
```

---

## Files

| File | Description |
|------|-------------|
| `occupancy_map.py` | Core `OccupancyMap` class — voxel grid, sensor updates, manifold extraction, path planning. Designed to be pulled into an existing control framework. |
| `simulator.py` | Simulation environment with synthetic terrain, DVL/sonar sensor models, and vehicle motion along the planned path. |
| `visualizer.py` | Browser-based real-time visualizer using WebSocket streaming. |

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

```python
from occupancy_map import OccupancyMap, OccupancyMapConfig
import numpy as np

# Configure
config = OccupancyMapConfig(
    dx=0.5,                 # X bin size (m)
    dz=0.25,                # Z bin size (m)
    horizon_fwd=15.0,       # Forward look-ahead (m)
    horizon_back=15.0,      # Backward look-behind (m)
    vehicle_length=2.0,     # Vehicle length for tail clearance (m)
    imaging_altitude=2.0,   # Target altitude above seafloor (m)
    cliff_standoff=2.0,     # Climb trigger distance before cliff (m)
)

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
