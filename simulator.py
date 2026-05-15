"""
AUV Obstacle Avoidance Simulator

A simple simulation loop that drives an AUV over synthetic terrain,
generates simulated DVL and forward sonar observations, feeds them
into the OccupancyMap, and logs state for visualization.

The simulator walks the vehicle along the path polyline at a constant
arc-length speed of 0.5 m/s, so the vehicle traces the planned path
faithfully — moving horizontally along flat segments and vertically
along cliff transitions.
"""

import math
import numpy as np
from collections import deque
from typing import Optional, Callable
from occupancy_map import (
    OccupancyMap, OccupancyMapConfig,
    DVLConfig, SonarConfig, AltimeterConfig,
    Pose, SensorType, DVLMeasurement, AltimeterMeasurement, SonarMeasurement,
    ObstacleMapper,
)

# ---------------------------------------------------------------------------
# Terrain generators
# ---------------------------------------------------------------------------



def default_terrain(world_x: float) -> float:
    """
    Synthetic seafloor terrain function.

    Returns depth (Z, positive down) at a given world X position.
    Includes gentle undulations with several cliff features.

    Args:
        world_x: World X coordinate (m).

    Returns:
        Seafloor depth (m).
    """
    d = 20.0
    d += 1.5 * np.sin(world_x * 0.15)
    d += 0.8 * np.sin(world_x * 0.4 + 1.2)
    d += 0.3 * np.sin(world_x * 0.9 + 2.5)

    cliffs = [
        {'x': 20, 'drop': 5.0, 'w': 12, 'r': 0.8, 'f': 1.5},
        {'x': 55, 'drop': 8.0, 'w': 18, 'r': 0.5, 'f': 2.0},
        {'x': 100, 'drop': 4.0, 'w': 10, 'r': 1.0, 'f': 1.0},
        {'x': 150, 'drop': 7.0, 'w': 20, 'r': 0.6, 'f': 1.8},
        {'x': 210, 'drop': 6.0, 'w': 14, 'r': 0.4, 'f': 1.2},
        {'x': 270, 'drop': 9.0, 'w': 22, 'r': 0.3, 'f': 2.5},
    ]

    for c in cliffs:
        cx, drop, w, rise, fall = c['x'], c['drop'], c['w'], c['r'], c['f']
        if cx < world_x < cx + rise:
            t = max(0, min(1, (world_x - cx) / rise))
            t = t * t * (3 - 2 * t)  # smoothstep
            d -= drop * t
        elif cx + rise <= world_x < cx + w:
            d -= drop
        elif cx + w <= world_x < cx + w + fall:
            t = max(0, min(1, (world_x - cx - w) / fall))
            t = t * t * (3 - 2 * t)
            d -= drop * (1 - t)

    return d


def make_sawtooth_terrain(
    slope_angle_deg: float = 45.0,
    amplitude: float = 10.0,
    base_depth: float = 20.0,
    flat_bottom: float = 0.0,
    reverse: bool = False,
) -> Callable[[float], float]:
    """
    Build a sawtooth slope terrain function.

    Normal direction (reverse=False):
        Each tooth rises linearly at *slope_angle_deg* from the trough up to
        the peak (peak-to-trough height = *amplitude*), then drops
        instantaneously (vertical cliff face) back to the trough.  An
        optional flat section of length *flat_bottom* metres is inserted at
        base depth between the drop and the next rising edge.

    Reversed direction (reverse=True):
        Each tooth starts with an instantaneous vertical rise (cliff face)
        from the trough to the peak, then descends linearly back to the
        trough.  The flat section (if any) follows the slope on the descent
        side.  This is the mirror image of the normal sawtooth and tests the
        vehicle approaching a sudden vertical wall (OBSTACLE_CLEAR) followed
        by a descent back to altitude following.

    Period in both cases:
        period = tooth_width + flat_bottom
        tooth_width = amplitude / tan(slope_angle_deg)

    At 0° the terrain is flat (no sawtooth).  At 90° the teeth are fully
    vertical (a series of instantaneous cliff steps); the tooth width is
    clamped to a minimum of 0.1 m so the period never collapses to zero.

    Args:
        slope_angle_deg: Slope angle of the gradual edge (degrees, 0–90).
        amplitude:       Peak-to-trough height (m).  Default 10 m.
        base_depth:      Seafloor depth at the trough (m, positive down).
        flat_bottom:     Length of flat terrain at base depth between teeth
                         (m, ≥ 0).  Default 0 m (original behaviour).
        reverse:         Flip tooth direction (default False).

    Returns:
        A callable ``terrain_fn(world_x) -> depth``.
    """
    if slope_angle_deg <= 0.0:
        return lambda x: base_depth

    slope_rad = np.radians(min(float(slope_angle_deg), 89.9))
    tooth_width = amplitude / np.tan(slope_rad)
    tooth_width = max(tooth_width, 0.1)   # avoid degenerate zero-width period
    flat_bottom = max(float(flat_bottom), 0.0)
    period = tooth_width + flat_bottom

    if not reverse:
        def terrain(world_x: float) -> float:
            x_in_period = world_x % period
            if x_in_period < tooth_width:
                # Gradual rising slope
                t = x_in_period / tooth_width
                return base_depth - t * amplitude
            # Flat bottom (also covers the instantaneous drop at end of tooth)
            return base_depth
    else:
        def terrain(world_x: float) -> float:  # type: ignore[misc]
            x_in_period = world_x % period
            if x_in_period < flat_bottom:
                # Flat bottom before the cliff face
                return base_depth
            # Gradual descending slope after the cliff face
            t = (x_in_period - flat_bottom) / tooth_width
            return (base_depth - amplitude) + t * amplitude

    direction = "reversed" if reverse else "normal"
    terrain.__name__ = (
        f"sawtooth(slope={slope_angle_deg:.1f}deg "
        f"amp={amplitude:.1f}m flat={flat_bottom:.1f}m "
        f"base={base_depth:.1f}m {direction})"
    )
    return terrain


# Registry of named terrain types for CLI / factory use
_TERRAIN_REGISTRY: dict[str, Callable] = {
    "default": lambda **_: default_terrain,
    "sawtooth": make_sawtooth_terrain,
}


def make_terrain(terrain_type: str = "default", **kwargs) -> Callable[[float], float]:
    """
    Return a terrain callable by name.

    Args:
        terrain_type: One of "default", "sawtooth".
        **kwargs:     Passed to the terrain factory (e.g. slope_angle_deg=30).

    Returns:
        A callable ``terrain_fn(world_x) -> depth``.

    Raises:
        ValueError: If *terrain_type* is not recognised.
    """
    if terrain_type not in _TERRAIN_REGISTRY:
        raise ValueError(
            f"Unknown terrain type '{terrain_type}'. "
            f"Available: {sorted(_TERRAIN_REGISTRY)}"
        )
    return _TERRAIN_REGISTRY[terrain_type](**kwargs)


class Simulator:
    """
    Drives the AUV simulation loop.

    The vehicle walks along the planned path polyline at constant arc-length
    speed. Sensors sample the terrain to build the occupancy map via the
    ObstacleMapper interface.
    """

    _STUCK_WINDOW_STEPS: int = 100
    _STUCK_PROGRESS_MIN: float = 0.3
    _OSCIL_WINDOW_STEPS: int = 12
    _OSCIL_MIN_FLIPS: int = 6
    _PERIODIC_PRINT_STEPS: int = 200

    def __init__(
        self,
        omap_config: Optional[OccupancyMapConfig] = None,
        dvl_config: Optional[DVLConfig] = None,
        sonar_config: Optional[SonarConfig] = None,
        altimeter_config: Optional[AltimeterConfig] = None,
        terrain_fn: Optional[Callable[[float], float]] = None,
        initial_depth: float = 0.0,
        dvl_hz: float = 8.0,
        altimeter_hz: float = 2.0,
        sonar_hz: float = 1.0,
        control_hz: float = 10.0,
        debug: bool = True,
    ):
        self.dvl = dvl_config or DVLConfig()
        self.sonar = sonar_config or SonarConfig()
        self.altimeter = altimeter_config or AltimeterConfig()
        self.mapper = ObstacleMapper(
            omap_config or OccupancyMapConfig(),
            self.dvl,
            self.sonar,
            self.altimeter,
        )
        self._dvl_period   = 1.0 / dvl_hz
        self._alt_period   = 1.0 / altimeter_hz
        self._sonar_period = 1.0 / sonar_hz
        self._ctrl_period  = 1.0 / control_hz
        self._dvl_last_t   = -self._dvl_period
        self._alt_last_t   = -self._alt_period
        self._sonar_last_t = -self._sonar_period
        self._ctrl_last_t  = -self._ctrl_period
        self.terrain_fn = terrain_fn or default_terrain
        self.debug = debug

        self.vehicle_x: float = 0.0
        self.vehicle_z: float = initial_depth
        self.time: float = 0.0

        # Cached control outputs — updated at control_hz, applied every dt
        self._ctrl_vx: float = 0.0   # forward speed (m/s)
        self._ctrl_vz: float = 0.0   # heave rate (m/s, positive = dive)

        self.mapper.reset(Pose(north=0.0, east=0.0, depth=initial_depth, heading=0.0))

        self.log: list = []

        self._step_count: int = 0
        self._x_history: deque = deque(maxlen=self._STUCK_WINDOW_STEPS)
        self._mode_history_full: deque = deque(maxlen=self._STUCK_WINDOW_STEPS)
        self._mode_history: deque = deque(maxlen=self._OSCIL_WINDOW_STEPS)
        self._last_stuck_report_x: float = -999.0

    def _simulate_dvl(self):
        """Ray-trace DVL beams against terrain; return (ranges, hit_surface)."""
        angles = self.dvl.beam_angles_rad
        ranges = np.zeros(len(angles))
        hit_surface = np.zeros(len(angles), dtype=bool)
        for i, ang in enumerate(angles):
            r = 0.1
            while r < self.dvl.max_range:
                hit_x = self.vehicle_x + np.sin(ang) * r
                hit_z = self.vehicle_z + np.cos(ang) * r
                if hit_z >= self.terrain_fn(hit_x):
                    ranges[i] = r
                    hit_surface[i] = True
                    break
                r += 0.1
            if not hit_surface[i]:
                ranges[i] = self.dvl.max_range
        return ranges, hit_surface

    def _simulate_altimeter(self):
        """Simulate altimeter: vertical range to seafloor directly below vehicle."""
        r = max(0.0, self.terrain_fn(self.vehicle_x) - self.vehicle_z)
        if r >= self.altimeter.max_range:
            return self.altimeter.max_range, False
        return r, True

    def _simulate_sonar(self):
        """Ray-trace forward sonar from vehicle nose; return (range_from_nose, hit)."""
        n_rays = 7
        angles = np.linspace(-self.sonar.half_angle_rad,
                             self.sonar.half_angle_rad, n_rays)
        nose_x = self.vehicle_x + self.mapper.omap.cfg.vehicle_length / 2.0
        min_range = self.sonar.max_range
        hit = False
        for ang in angles:
            r = 0.2
            while r < self.sonar.max_range:
                hit_x = nose_x + r * np.cos(ang)
                hit_z = self.vehicle_z + r * np.sin(ang)
                if hit_z >= self.terrain_fn(hit_x):
                    noisy_r = r + np.random.normal(0, self.sonar.noise_std)
                    if noisy_r < min_range:
                        min_range = max(0.1, noisy_r)
                    hit = True
                    break
                r += 0.2
        return min_range, hit

    def _cmd_depth_at_x(self, world_x: float) -> float:
        """Linearly interpolate mapper.omap.cmd_depth at world_x."""
        omap = self.mapper.omap
        if omap.nx < 2:
            return self.vehicle_z
        rel = (world_x - omap.grid_origin_x) / omap.cfg.dx
        ix_low = int(np.floor(rel))
        ix_high = ix_low + 1
        if ix_low < 0:
            ix_low, ix_high = 0, 1
        if ix_high >= omap.nx:
            ix_low, ix_high = omap.nx - 2, omap.nx - 1
        z_low = omap.cmd_depth[ix_low]
        z_high = omap.cmd_depth[ix_high]
        if np.isnan(z_low) and np.isnan(z_high):
            return self.vehicle_z
        if np.isnan(z_low):
            return float(z_high)
        if np.isnan(z_high):
            return float(z_low)
        t = max(0.0, min(1.0, rel - ix_low))
        return float(z_low + t * (z_high - z_low))

    def step(self, dt: float):
        """Advance simulation by dt seconds.

        Each step: kinematics are integrated first so the vehicle pose is
        current, then sensor callbacks fire with that accurate pose, then the
        control loop ticks (at control_hz) and updates the cached velocity
        commands that will drive the next steps.

        Order:
          1. Integrate kinematics → updated (x, z, t).
          2. Build pose from updated position.
          3. Fire sensors (DVL / altimeter / sonar) with current pose.
          4. Control tick (10 Hz): pass pose to mapper, query altitude + command, update vx/vz cache.
        """
        # 1. Integrate kinematics with current cached velocity commands
        self.vehicle_z += self._ctrl_vz * dt
        self.vehicle_x += self._ctrl_vx * dt
        self.vehicle_z = max(0.0, self.vehicle_z)
        self.time += dt

        # 2. Pose from updated position
        pose = Pose(north=self.vehicle_x, east=0.0, depth=self.vehicle_z, heading=0.0)

        # 3. Sensor callbacks with accurate pose
        if self.time >= self._dvl_last_t + self._dvl_period:
            dvl_ranges, dvl_hits = self._simulate_dvl()
            self.mapper.update_sensor(SensorType.DVL, DVLMeasurement(dvl_ranges, dvl_hits), pose)
            self._dvl_last_t += self._dvl_period

        if self.time >= self._alt_last_t + self._alt_period:
            alt_range, alt_hit = self._simulate_altimeter()
            self.mapper.update_sensor(SensorType.ALTIMETER, AltimeterMeasurement(alt_range, alt_hit), pose)
            self._alt_last_t += self._alt_period

        if self.time >= self._sonar_last_t + self._sonar_period:
            sonar_range, sonar_hit = self._simulate_sonar()
            self.mapper.update_sensor(SensorType.SONAR, SonarMeasurement(sonar_range, sonar_hit), pose)
            self._sonar_last_t += self._sonar_period

        # 4. Control tick — runs at control_hz, independent of sensor rates
        if self.time >= self._ctrl_last_t + self._ctrl_period:
            self.mapper.update_pose(pose)
            ctrl_alt = self.mapper.get_altitude()
            ctrl_cmd = self.mapper.get_control()
            c = self.mapper.omap.cfg
            self._ctrl_vx = ctrl_cmd.vx
            if ctrl_cmd.vertical_mode == 'ALT_FOLLOW':
                if not np.isnan(ctrl_alt):
                    # Positive vz = dive; positive when alt > target (need to descend)
                    self._ctrl_vz = float(np.clip(
                        ctrl_alt - ctrl_cmd.vertical_target,
                        -c.vertical_speed, c.vertical_speed,
                    ))
                else:
                    self._ctrl_vz = 0.0
            else:  # DEPTH_HOLD
                self._ctrl_vz = float(np.clip(
                    ctrl_cmd.vertical_target - self.vehicle_z,
                    -c.vertical_speed, c.vertical_speed,
                ))
            self._ctrl_last_t += self._ctrl_period

        cmd_depth = self.mapper.omap.get_commanded_depth_at_vehicle()

        terrain_z = self.terrain_fn(self.vehicle_x)
        altitude = terrain_z - self.vehicle_z
        state = {
            'time': self.time,
            'vehicle_x': self.vehicle_x,
            'vehicle_z': self.vehicle_z,
            'terrain_z': terrain_z,
            'altitude': altitude,
            'cmd_depth': cmd_depth if not np.isnan(cmd_depth) else self.vehicle_z,
        }
        self.log.append(state)

        if self.debug:
            self._debug_check(terrain_z, altitude)

        return state

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _debug_check(self, terrain_z: float, altitude: float):
        self._step_count += 1
        self._x_history.append(self.vehicle_x)
        omap = self.mapper.omap
        self._mode_history.append(omap.control_mode)
        self._mode_history_full.append(omap.control_mode)

        if self._step_count % self._PERIODIC_PRINT_STEPS == 0:
            alt_reading = self.mapper.get_altitude()
            alt_s = f"{alt_reading:5.2f}" if not np.isnan(alt_reading) else "  nan"
            print(
                f"[T={self.time:7.1f}s  X={self.vehicle_x:7.1f}m  Z={self.vehicle_z:6.2f}m  "
                f"alt={altitude:5.2f}m  terrain={terrain_z:6.2f}m  "
                f"mode={omap.control_mode:<15} sensor_alt={alt_s}m]"
            )

        if len(self._x_history) < self._STUCK_WINDOW_STEPS:
            return

        progress = self._x_history[-1] - self._x_history[0]
        all_alt_follow = (
            len(self._mode_history_full) == self._STUCK_WINDOW_STEPS
            and all(m == "ALT_FOLLOW" for m in self._mode_history_full)
        )
        is_stuck = all_alt_follow and progress < self._STUCK_PROGRESS_MIN
        flips = sum(
            1 for i in range(1, len(self._mode_history))
            if self._mode_history[i] != self._mode_history[i - 1]
        )
        is_oscillating = flips >= self._OSCIL_MIN_FLIPS
        too_close = abs(self.vehicle_x - self._last_stuck_report_x) < 2.0

        if (is_stuck or is_oscillating) and not too_close:
            self._last_stuck_report_x = self.vehicle_x
            reason = []
            if is_stuck:
                reason.append(
                    f"ALT_FOLLOW with <{self._STUCK_PROGRESS_MIN}m progress "
                    f"({progress:.3f}m in {self._STUCK_WINDOW_STEPS} steps)"
                )
            if is_oscillating:
                reason.append(f"mode oscillation ({flips} flips in {self._OSCIL_WINDOW_STEPS} steps)")
            print(
                f"\n{'='*70}\n"
                f"  VEHICLE STUCK/OSCILLATING  T={self.time:.1f}s  "
                f"X={self.vehicle_x:.3f}m  Z={self.vehicle_z:.3f}m\n"
                f"  Reason: {'; '.join(reason)}\n"
                f"  Terrain: {terrain_z:.3f}m  Altitude: {altitude:.3f}m  "
                f"Target alt: {omap.cfg.imaging_altitude:.2f}m\n"
                f"  Mode history (last {self._OSCIL_WINDOW_STEPS}): "
                f"{' '.join(m[0] for m in self._mode_history)}\n"
                + omap.get_debug_summary(self.vehicle_z) +
                f"\n{'='*70}"
            )

    def run(self, duration: float, dt: float = 0.1) -> list:
        """Run simulation for a given duration."""
        for _ in range(int(np.ceil(duration / dt))):
            self.step(dt)
        return self.log


# ===========================================================================
# 3D Terrain Functions
# ===========================================================================

def make_terrain_3d_flat(depth: float = 20.0) -> Callable[[float, float], float]:
    """Flat seafloor at constant depth."""
    def terrain(x: float, y: float) -> float:
        return depth
    terrain.__name__ = "flat_3d"
    return terrain


def make_terrain_3d_seamount(
    cx: float = 80.0,
    cy: float = 0.0,
    radius: float = 30.0,
    height: float = 14.0,
    base_depth: float = 22.0,
) -> Callable[[float, float], float]:
    """Steep-sided seamount with a flat top.

    Uses a square-root radial profile ``1 - √(r/R)`` which produces nearly
    vertical walls near the base and a broad, flat summit — more cliff-like
    than the original parabolic dome.
    """
    def terrain(x: float, y: float) -> float:
        r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        rise = height * max(0.0, 1.0 - np.sqrt(r / radius))  # steep sides
        return base_depth - rise
    terrain.__name__ = "seamount_3d"
    return terrain


def make_terrain_3d_ridge(
    ridge_heading_deg: float = 90.0,
    amplitude: float = 8.0,
    period: float = 60.0,
    base_depth: float = 20.0,
) -> Callable[[float, float], float]:
    """Ridge running parallel to *ridge_heading_deg* with steep sides.

    Uses a clipped cosine profile ``max(0, cos(2πs/period))`` which gives
    sharp, steep-sided peaks separated by flat troughs — more cliff-like than
    the original raised-cosine (which was always positive and had gentle sides).
    """
    perp = np.radians(ridge_heading_deg + 90.0)

    def terrain(x: float, y: float) -> float:
        s    = x * np.cos(perp) + y * np.sin(perp)
        rise = amplitude * max(0.0, np.cos(2.0 * np.pi * s / period))
        return base_depth - rise
    terrain.__name__ = "ridge_3d"
    return terrain


def make_terrain_3d_canyon(
    travel_heading_deg: float = 0.0,
    center_offset: float = 0.0,
    width: float = 40.0,
    extra_depth: float = 8.0,
    base_depth: float = 20.0,
) -> Callable[[float, float], float]:
    """Canyon (trench) running parallel to *travel_heading_deg*.

    *center_offset* shifts the canyon centre laterally from the travel line.
    """
    h = np.radians(travel_heading_deg)
    perp = h + np.pi / 2.0

    def terrain(x: float, y: float) -> float:
        d = x * np.cos(perp) + y * np.sin(perp) - center_offset
        if abs(d) < width / 2.0:
            return base_depth + extra_depth * (1.0 - (2.0 * d / width) ** 2)
        return base_depth
    terrain.__name__ = "canyon_3d"
    return terrain


def make_terrain_3d_slope(
    angle_deg: float = 5.0,
    slope_heading_deg: float = 0.0,
    base_depth: float = 10.0,
) -> Callable[[float, float], float]:
    """Uniformly sloping seafloor rising in the *slope_heading_deg* direction."""
    h = np.radians(slope_heading_deg)
    slope = np.tan(np.radians(angle_deg))

    def terrain(x: float, y: float) -> float:
        s = x * np.cos(h) + y * np.sin(h)
        return base_depth + slope * s
    terrain.__name__ = "slope_3d"
    return terrain


def extrude_terrain_3d(
    terrain_2d_fn: Callable[[float], float],
    extrude_heading_deg: float = 0.0,
) -> Callable[[float, float], float]:
    """Wrap a 2D terrain function ``f(s) → z`` as a 3D terrain ``f(x, y) → z``
    by projecting (x, y) onto *extrude_heading_deg*."""
    h = np.radians(extrude_heading_deg)

    def terrain(x: float, y: float) -> float:
        s = x * np.cos(h) + y * np.sin(h)
        return terrain_2d_fn(s)
    terrain.__name__ = getattr(terrain_2d_fn, '__name__', 'extruded') + "_3d"
    return terrain


def make_terrain_3d_sawtooth(
    slope_angle_deg: float = 45.0,
    amplitude: float = 10.0,
    base_depth: float = 40.0,
    flat_bottom: float = 0.0,
    reverse: bool = False,
    orientation_deg: float = 0.0,
) -> Callable[[float, float], float]:
    """3-D sawtooth terrain.

    The sawtooth pattern from :func:`make_sawtooth_terrain` is extruded
    perpendicularly to *orientation_deg*, so the teeth progress in the
    *orientation_deg* direction.  At 0° (default) the teeth are identical to
    the 2-D sawtooth along the X axis; at 90° they run along the Y axis.

    Args:
        slope_angle_deg: Slope angle of the gradual edge (degrees, 0–90).
        amplitude:       Peak-to-trough height (m).
        base_depth:      Seafloor depth at the trough (m, positive down).
                         Default 40 m ensures the peak (base_depth − amplitude)
                         stays well below the surface even with large amplitudes.
        flat_bottom:     Flat terrain length between teeth (m).
        reverse:         Flip tooth direction (vertical rise then gradual down).
        orientation_deg: Direction teeth progress in (degrees, 0 = +X axis).
    """
    terrain_2d = make_sawtooth_terrain(
        slope_angle_deg=slope_angle_deg,
        amplitude=amplitude,
        base_depth=base_depth,
        flat_bottom=flat_bottom,
        reverse=reverse,
    )
    fn = extrude_terrain_3d(terrain_2d, extrude_heading_deg=orientation_deg)
    direction = "rev" if reverse else "fwd"
    fn.__name__ = (
        f"sawtooth3d(slope={slope_angle_deg:.0f}° amp={amplitude:.0f}m "
        f"flat={flat_bottom:.0f}m orient={orientation_deg:.0f}° {direction})"
    )
    return fn


def make_terrain_3d_random_reef(
    seed: int | None = None,
    base_depth: float = 40.5,
    min_depth: float = 8.5,
    max_depth: float = 98.0,
    reef_grid_dx: float = 0.55,
    reef_grid_xlim: tuple[float, float] | None = None,
    reef_grid_ylim: tuple[float, float] | None = None,
) -> Callable[[float, float], float]:
    """Procedural reef-like bathymetry: bommies, terraces / cliff-like faults,
    low-frequency swell, and high rugosity (local rises and falls of typically
    2-3 m over horizontal scales of roughly 1-5 m mixed with larger structure).

    Each call draws a fresh realisation unless *seed* is fixed. Typical use is
    the ``default`` 3-D terrain preset for varied obstacle-avoidance stress.

    Output depth ``z`` increases downward (positive into the water column).
    Shallow reefs may approach *min_depth*; deep gouges clip at *max_depth*.

    For simulation speed the field is **rasterised once** on a regular XY grid
    (defaults cover lawnmower-scale missions) and evaluated at runtime via
    bilinear interpolation.  Finer *reef_grid_dx* or wider ``reef_grid_*lim``
    improves fidelity outside the window at some build cost.
    """
    rng = np.random.default_rng(seed)

    x0, x1 = reef_grid_xlim if reef_grid_xlim is not None else (-52.0, 268.0)
    y0, y1 = reef_grid_ylim if reef_grid_ylim is not None else (-125.0, 125.0)
    dxg = max(0.12, float(reef_grid_dx))

    nx = max(3, int(np.ceil((x1 - x0) / dxg)) + 1)
    ny = max(3, int(np.ceil((y1 - y0) / dxg)) + 1)
    xs = np.linspace(x0, x1, nx, dtype=np.float64)
    ys = np.linspace(y0, y1, ny, dtype=np.float64)
    xstep = float(xs[1] - xs[0]) if nx > 1 else 1.0
    ystep = float(ys[1] - ys[0]) if ny > 1 else 1.0
    X, Y = np.meshgrid(xs, ys, indexing='ij')
    Z = np.full(X.shape, float(base_depth), dtype=np.float64)

    # ------- Long swell (overall slope envelope) --------
    swell_k = rng.uniform(np.pi / 220.0, np.pi / 75.0, size=(4, 2))
    swell_a = rng.uniform(2.0, 7.5, size=4)
    swell_p = rng.uniform(0.0, 2.0 * np.pi, size=4)

    mid_k = rng.uniform(np.pi / 76.0, np.pi / 22.0, size=(7, 2))
    mid_a = rng.uniform(1.5, 5.8, size=7)
    mid_p = rng.uniform(0.0, 2.0 * np.pi, size=7)

    # ------- Rugosity: multiple bands for reef-like texture --------
    nk = int(rng.integers(28, 46))
    L = rng.uniform(1.0, 6.8, size=nk)
    ang = rng.uniform(0.0, 2.0 * np.pi, size=nk)
    rk_kx = (2.0 * np.pi / L) * np.cos(ang)
    rk_ky = (2.0 * np.pi / L) * np.sin(ang)
    rk_a = rng.uniform(0.35, 2.05, size=nk) * 0.82
    rk_p = rng.uniform(0.0, 2.0 * np.pi, size=nk)

    nk2 = int(rng.integers(18, 30))
    L2 = rng.uniform(1.05, 3.95, size=nk2)
    ang2 = rng.uniform(0.0, 2.0 * np.pi, size=nk2)
    rk2_kx = (2.0 * np.pi / L2) * np.cos(ang2)
    rk2_ky = (2.0 * np.pi / L2) * np.sin(ang2)
    rk_a2 = rng.uniform(0.27, 1.45, size=nk2) * 0.78
    rk_p2 = rng.uniform(0.0, 2.0 * np.pi, size=nk2)

    # ------- Bommies (Gaussian mounds, shallow apex) --------
    n_bom = int(rng.integers(21, 40))
    bx = rng.uniform(-25.0, 220.0, size=n_bom)
    by = rng.uniform(-90.0, 90.0, size=n_bom)
    b_wx = rng.uniform(2.2, 17.5, size=n_bom)
    b_wy = rng.uniform(2.2, 14.8, size=n_bom)
    b_rot = rng.uniform(0.0, np.pi, size=n_bom)
    b_h = rng.uniform(2.0, 16.8, size=n_bom)

    # ------- Inverse-Gaussian “bowls” / drop-offs behind bommies --------
    n_hole = int(rng.integers(6, 15))
    hx = rng.uniform(-20.0, 210.0, size=n_hole)
    hy = rng.uniform(-75.0, 75.0, size=n_hole)
    h_w = rng.uniform(5.6, 32.0, size=n_hole)
    h_d = rng.uniform(2.0, 12.8, size=n_hole)

    # ------- Near-sheer terrace / cliff faults across the scene --------
    n_fault = int(rng.integers(6, 13))
    fx0 = rng.uniform(-10.0, 190.0, size=n_fault)
    fy0 = rng.uniform(-70.0, 70.0, size=n_fault)
    fh = rng.uniform(0.0, np.pi * 2.0, size=n_fault)
    f_nx = np.cos(fh)
    f_ny = np.sin(fh)
    f_drop = rng.uniform(3.5, 12.9, size=n_fault)
    f_sharp = rng.uniform(22.0, 120.0, size=n_fault)
    f_sg = rng.uniform(0.0, 3.14159, size=n_fault)

    # --- Vector accumulation on grid ---
    for i in range(4):
        Z += swell_a[i] * np.sin(
            swell_k[i, 0] * X + swell_k[i, 1] * Y + swell_p[i])
    for i in range(7):
        Z += mid_a[i] * np.sin(
            mid_k[i, 0] * X + mid_k[i, 1] * Y + mid_p[i])

    for i in range(nk):
        Z += rk_a[i] * np.sin(
            rk_kx[i] * X + rk_ky[i] * Y + rk_p[i])
    for i in range(nk2):
        Z += rk_a2[i] * np.sin(
            rk2_kx[i] * X + rk2_ky[i] * Y + rk_p2[i])

    for i in range(n_bom):
        ct, st = float(np.cos(b_rot[i])), float(np.sin(b_rot[i]))
        dxm = X - bx[i]
        dym = Y - by[i]
        xr = ct * dxm - st * dym
        yr = st * dxm + ct * dym
        ex = (xr ** 2 / (b_wx[i] ** 2 + 0.06)
              + yr ** 2 / (b_wy[i] ** 2 + 0.06))
        Z -= b_h[i] * np.exp(-0.92 * np.minimum(ex, 48.0))

    for i in range(n_hole):
        r2 = (X - hx[i]) ** 2 + (Y - hy[i]) ** 2
        Z += h_d[i] * np.exp(-0.53 * r2 / (h_w[i] ** 2 + 0.15))

    for i in range(n_fault):
        t = ((X - fx0[i]) * f_nx[i] + (Y - fy0[i]) * f_ny[i]
             + math.sin(float(f_sg[i])))
        arg = np.clip(-float(f_sharp[i]) * t, -708.0, 708.0)
        sig = 1.0 / (1.0 + np.exp(arg))
        Z += float(f_drop[i]) * (sig - 0.5)

    np.clip(Z, min_depth, max_depth, out=Z)

    def terrain(x: float, y: float) -> float:
        xc = float(np.clip(x, xs[0], xs[-1]))
        yc = float(np.clip(y, ys[0], ys[-1]))
        xf = (xc - xs[0]) / xstep
        yf = (yc - ys[0]) / ystep
        i0 = int(np.floor(xf))
        j0 = int(np.floor(yf))
        i0 = min(max(i0, 0), nx - 2)
        j0 = min(max(j0, 0), ny - 2)
        tx = min(max(float(xf - i0), 0.0), 1.0)
        ty = min(max(float(yf - j0), 0.0), 1.0)
        z = (
            (1.0 - tx) * (1.0 - ty) * Z[i0, j0]
            + tx * (1.0 - ty) * Z[i0 + 1, j0]
            + (1.0 - tx) * ty * Z[i0, j0 + 1]
            + tx * ty * Z[i0 + 1, j0 + 1]
        )
        return float(z)

    terrain.__name__ = "random_reef_3d"
    terrain._reef_grid_shape = (nx, ny)  # debug / tuning
    return terrain


def make_terrain_3d_default(base_depth: float = 30.0) -> Callable[[float, float], float]:
    """Default 3D test terrain: a seamount centred 40 m ahead with gentle
    background ridges (XY scale halved; vertical scale unchanged)."""
    seamount = make_terrain_3d_seamount(cx=40, cy=0, radius=15, height=20,
                                        base_depth=base_depth)
    ridge = make_terrain_3d_ridge(ridge_heading_deg=90, amplitude=10, period=25,
                                   base_depth=base_depth)

    def terrain(x: float, y: float) -> float:
        return min(seamount(x, y), ridge(x, y))
    terrain.__name__ = "default_3d"
    return terrain


_TERRAIN_3D_REGISTRY: dict = {
    'flat':      make_terrain_3d_flat,
    'seamount':  make_terrain_3d_seamount,
    'ridge':     make_terrain_3d_ridge,
    'canyon':    make_terrain_3d_canyon,
    'slope':     make_terrain_3d_slope,
    'sawtooth':  make_terrain_3d_sawtooth,
    'classic':   make_terrain_3d_default,
    'default':   make_terrain_3d_random_reef,
}


def make_terrain_3d(terrain_type: str = 'default', **kwargs) -> Callable[[float, float], float]:
    """Return a 3D terrain callable ``f(x, y) → depth`` by name."""
    if terrain_type not in _TERRAIN_3D_REGISTRY:
        raise ValueError(
            f"Unknown 3D terrain type '{terrain_type}'. "
            f"Available: {sorted(_TERRAIN_3D_REGISTRY)}"
        )
    return _TERRAIN_3D_REGISTRY[terrain_type](**kwargs)


# ===========================================================================
# 3D Trajectories
# ===========================================================================

class Trajectory3D:
    """Abstract base for 3D trajectory specifications.

    A trajectory maps cumulative arc-length (distance travelled) and the
    vehicle's current world (x, y) to the desired heading (radians, 0 = +X).
    """

    def heading_at(self, arc_length: float, x: float, y: float) -> float:
        raise NotImplementedError


class StraightTrajectory3D(Trajectory3D):
    """Constant heading."""

    def __init__(self, heading_deg: float = 0.0):
        self._h = np.radians(heading_deg)

    def heading_at(self, arc_length: float, x: float, y: float) -> float:
        return self._h


class ArcTrajectory3D(Trajectory3D):
    """Circular arc — constant turn rate.

    Args:
        heading_start_deg: Initial heading in degrees.
        radius: Turn radius in metres (larger = gentler turn).
        direction: ``'left'`` (CCW, +yaw) or ``'right'`` (CW, -yaw).
    """

    def __init__(
        self,
        heading_start_deg: float = 0.0,
        radius: float = 50.0,
        direction: str = 'left',
    ):
        self._h0 = np.radians(heading_start_deg)
        self._r = max(1.0, radius)
        self._sign = 1.0 if direction.lower() in ('left', 'l', 'ccw') else -1.0

    def heading_at(self, arc_length: float, x: float, y: float) -> float:
        return self._h0 + self._sign * arc_length / self._r


class WaypointTrajectory3D(Trajectory3D):
    """Navigate through a sequence of (x, y) XY waypoints.

    The vehicle steers toward the closest uncompleted waypoint.  A waypoint is
    considered reached when the vehicle is within *lookahead* metres of it.
    """

    def __init__(self, waypoints: list, lookahead: float = 4.0):
        self._wp = list(waypoints)
        self._lookahead = lookahead
        self._target_i = 0

    def heading_at(self, arc_length: float, x: float, y: float) -> float:
        while (self._target_i < len(self._wp) - 1 and
               np.hypot(x - self._wp[self._target_i][0],
                        y - self._wp[self._target_i][1]) < self._lookahead):
            self._target_i += 1
        tx, ty = self._wp[min(self._target_i, len(self._wp) - 1)]
        return float(np.arctan2(ty - y, tx - x))


class SegmentedTrajectory3D(Trajectory3D):
    """Concatenated trajectory segments.

    Args:
        segments: List of ``(distance_m, Trajectory3D)`` tuples.  The vehicle
            follows each sub-trajectory for the specified distance before
            switching to the next.
    """

    def __init__(self, segments: list):
        self._segments = segments
        self._cumulative = [0.0]
        for dist, _ in segments:
            self._cumulative.append(self._cumulative[-1] + float(dist))

    @property
    def total_arc_length(self) -> float:
        """Total arc-length of all segments."""
        return self._cumulative[-1]

    def heading_at(self, arc_length: float, x: float, y: float) -> float:
        for i in range(len(self._segments)):
            if arc_length <= self._cumulative[i + 1] or i == len(self._segments) - 1:
                seg_start = self._cumulative[i]
                return self._segments[i][1].heading_at(arc_length - seg_start, x, y)
        return self._segments[-1][1].heading_at(0.0, x, y)


# ===========================================================================
# Lawnmower mission builder
# ===========================================================================

def _integrate_trajectory_path(
    traj: 'SegmentedTrajectory3D',
    start_x: float = 0.0,
    start_y: float = 0.0,
    step_ds: float = 0.5,
) -> list:
    """Integrate a SegmentedTrajectory3D into a list of (x, y) world points.

    Used to generate a smooth rendered mission path from any trajectory
    (including ones with arc corners).
    """
    total = traj.total_arc_length
    x, y  = float(start_x), float(start_y)
    path  = [(x, y)]
    s     = 0.0
    while s < total:
        ds = min(step_ds, total - s)
        h  = traj.heading_at(s, x, y)
        x += ds * np.cos(h)
        y += ds * np.sin(h)
        s += ds
        path.append((x, y))
    return path


def make_lawnmower_trajectory(
    leg_length: float = 20.0,
    spacing: float = 7.0,
    n_legs: int = 20,
    orientation_deg: float = 0.0,
    start_x: float = 0.0,
    start_y: float = 0.0,
    turn_rate: float = 0.25,
    survey_speed: float = 0.5,
) -> tuple:
    """Build a lawnmower survey trajectory with smooth arc corners.

    Each leg is a straight run aligned with *orientation_deg*.  At the end of
    each leg the vehicle turns through 90° using a circular arc whose radius
    is derived from *survey_speed* / *turn_rate*.  Setting *turn_rate* = 0
    (default) produces instant square corners (original behaviour).

    Args:
        leg_length:      Length of each parallel leg in metres.
        spacing:         Cross-track spacing between adjacent legs in metres.
        n_legs:          Number of parallel legs.
        orientation_deg: Heading of the first (and all odd) legs, degrees CCW
                         from +X.
        start_x / start_y: Starting world position.
        turn_rate:       Corner turn rate in rad/s (0 = instant square turns).
        survey_speed:    Vehicle survey speed in m/s; used only to compute the
                         arc radius when *turn_rate* > 0.

    Returns:
        ``(SegmentedTrajectory3D, path_xy)`` where *path_xy* is a list of
        ``(x, y)`` world-coordinate points traced by the vehicle (smooth curve
        through arc corners when turn_rate > 0; corner waypoints otherwise).
    """
    # --- Turn geometry -------------------------------------------------------
    if turn_rate > 0.0:
        radius      = survey_speed / turn_rate          # e.g. 0.5/0.25 = 2 m
        arc_len     = radius * np.pi / 2.0              # 90° arc arc-length
        # Shorten straight portions so the arc corner fits within the pattern
        leg_straight   = max(0.1, leg_length - radius)
        lat_straight   = max(0.0, spacing - 2.0 * radius)
    else:
        radius      = 0.0
        arc_len     = 0.0
        leg_straight   = leg_length
        lat_straight   = spacing

    # --- Heading constants ---------------------------------------------------
    fwd_hdg  = orientation_deg            # even legs
    rev_hdg  = orientation_deg + 180.0    # odd legs
    lat_hdg  = orientation_deg + 90.0     # cross-track direction

    segments: list = []

    for i in range(n_legs):
        leg_hdg = fwd_hdg if i % 2 == 0 else rev_hdg

        # Straight portion of this leg (full length for the last leg)
        if i < n_legs - 1:
            segments.append((leg_straight, StraightTrajectory3D(leg_hdg)))
        else:
            segments.append((leg_length, StraightTrajectory3D(leg_hdg)))
            break  # no corner after the last leg

        # Corner: turn from leg_hdg → lat_hdg → next_leg_hdg
        # Even legs turn LEFT (CCW), odd legs turn RIGHT (CW)
        turn_dir = 'left' if i % 2 == 0 else 'right'

        if arc_len > 0.0:
            # First arc: leg_hdg → lat_hdg
            segments.append((arc_len, ArcTrajectory3D(leg_hdg,  radius, turn_dir)))
            # Cross-track straight
            if lat_straight > 0.0:
                segments.append((lat_straight, StraightTrajectory3D(lat_hdg)))
            # Second arc: lat_hdg → next_leg_hdg
            segments.append((arc_len, ArcTrajectory3D(lat_hdg, radius, turn_dir)))
        else:
            # Instant 90° heading changes — square corners
            segments.append((spacing, StraightTrajectory3D(lat_hdg)))

    traj = SegmentedTrajectory3D(segments)

    if turn_rate > 0.0:
        # Integrate the arc-curved trajectory to get a smooth rendered path
        path = _integrate_trajectory_path(traj, start_x, start_y, step_ds=0.5)
    else:
        # Compute exact corner waypoints analytically for clean square rendering
        h_rad   = np.radians(orientation_deg)
        cos_h   = np.cos(h_rad)
        sin_h   = np.sin(h_rad)
        cos_lat = np.cos(h_rad + np.pi / 2.0)
        sin_lat = np.sin(h_rad + np.pi / 2.0)
        x, y    = float(start_x), float(start_y)
        path    = [(x, y)]
        for i in range(n_legs):
            dx_leg = (cos_h if i % 2 == 0 else -cos_h) * leg_length
            dy_leg = (sin_h if i % 2 == 0 else -sin_h) * leg_length
            x += dx_leg;  y += dy_leg
            path.append((x, y))
            if i < n_legs - 1:
                x += cos_lat * spacing;  y += sin_lat * spacing
                path.append((x, y))

    return traj, path


# ===========================================================================
# Simulator3D
# ===========================================================================

class Simulator3D:
    """3D AUV simulation.

    The vehicle follows an XY trajectory while depth is controlled by the
    same 2D obstacle avoidance system as :class:`Simulator`. Sensor beams
    (DVL, sonar) are cast in 3D against the terrain function and submitted
    via the ObstacleMapper interface — the mapper projects them onto the
    along-heading axis internally.

    Args:
        terrain_fn:  ``f(x, y) → depth`` callable.
        trajectory:  :class:`Trajectory3D` instance or ``None`` for straight.
        initial_x / initial_y / initial_heading_deg:  Starting XY position and heading.
        initial_depth:  Starting depth (m, positive down).
    """

    _STUCK_WINDOW_STEPS:    int   = 100
    _STUCK_PROGRESS_MIN:    float = 0.3
    _OSCIL_WINDOW_STEPS:    int   = 12
    _OSCIL_MIN_FLIPS:       int   = 6
    _PERIODIC_PRINT_STEPS:  int   = 200

    def __init__(
        self,
        omap_config: Optional[OccupancyMapConfig] = None,
        dvl_config:  Optional[DVLConfig]           = None,
        sonar_config: Optional[SonarConfig]        = None,
        altimeter_config: Optional[AltimeterConfig] = None,
        terrain_fn:  Optional[Callable]            = None,
        trajectory:  Optional[Trajectory3D]        = None,
        initial_x:   float = 0.0,
        initial_y:   float = 0.0,
        initial_depth: float = 0.0,
        initial_heading_deg: float = 0.0,
        dvl_hz: float = 8.0,
        altimeter_hz: float = 2.0,
        sonar_hz: float = 1.0,
        control_hz: float = 10.0,
        debug: bool = True,
    ):
        self.dvl       = dvl_config       or DVLConfig()
        self.sonar     = sonar_config     or SonarConfig()
        self.altimeter = altimeter_config or AltimeterConfig()
        self.mapper = ObstacleMapper(
            omap_config or OccupancyMapConfig(),
            self.dvl,
            self.sonar,
            self.altimeter,
        )
        self._dvl_period   = 1.0 / dvl_hz
        self._alt_period   = 1.0 / altimeter_hz
        self._sonar_period = 1.0 / sonar_hz
        self._ctrl_period  = 1.0 / control_hz
        self._dvl_last_t   = -self._dvl_period
        self._alt_last_t   = -self._alt_period
        self._sonar_last_t = -self._sonar_period
        self._ctrl_last_t  = -self._ctrl_period
        self.terrain_fn = terrain_fn or make_terrain_3d('default')
        self.trajectory = trajectory or StraightTrajectory3D(initial_heading_deg)
        self.debug = debug

        self.vehicle_x: float = initial_x
        self.vehicle_y: float = initial_y
        self.vehicle_z: float = initial_depth
        self.vehicle_heading: float = np.radians(initial_heading_deg)
        self.arc_length: float = 0.0
        self.time: float = 0.0

        # Cached control outputs — updated at control_hz, applied every dt
        self._ctrl_vx: float = 0.0   # forward speed (m/s)
        self._ctrl_vz: float = 0.0   # heave rate (m/s, positive = dive)

        self.mapper.reset(Pose(
            north=initial_x, east=initial_y,
            depth=initial_depth, heading=self.vehicle_heading,
        ))

        self.log: list = []
        self._step_count:       int   = 0
        self._x_history:        deque = deque(maxlen=self._STUCK_WINDOW_STEPS)
        self._mode_history:     deque = deque(maxlen=self._OSCIL_WINDOW_STEPS)
        self._mode_history_full:deque = deque(maxlen=self._STUCK_WINDOW_STEPS)
        self._last_stuck_report_x: float = -999.0

        self.xy_trail: list = []

        n_beams = len(self.dvl.beam_angles_rad)
        self.dvl_hit_xy:   list  = [None] * n_beams
        self.sonar_hit_xy: tuple | None = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def omap(self):
        """OccupancyMap — exposed for visualizer and debug access."""
        return self.mapper.omap

    @property
    def _arc_local(self) -> float:
        """Along-track distance traveled (visualizer compatibility)."""
        return self.arc_length

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _terrain_at(self, x: float, y: float) -> float:
        try:
            return float(self.terrain_fn(x, y))
        except Exception:
            return 20.0

    # ------------------------------------------------------------------
    # Sensor simulation (3D)
    # ------------------------------------------------------------------

    def _simulate_dvl(self):
        """Ray-trace DVL beams in 3D. Returns (ranges, hit_surface, hit_xy)."""
        h = self.vehicle_heading
        cos_h, sin_h = np.cos(h), np.sin(h)
        beam_dirs    = self.dvl.beam_directions_3d
        n_beams      = len(beam_dirs)
        ranges       = np.zeros(n_beams)
        hit_surface  = np.zeros(n_beams, dtype=bool)
        hit_xy       = [None] * n_beams

        for i, (fwd, stbd, down) in enumerate(beam_dirs):
            r = 0.1
            while r < self.dvl.max_range:
                hx = self.vehicle_x + (fwd * cos_h - stbd * sin_h) * r
                hy = self.vehicle_y + (fwd * sin_h + stbd * cos_h) * r
                hz = self.vehicle_z + down * r
                if hz >= self._terrain_at(hx, hy):
                    ranges[i]      = r
                    hit_surface[i] = True
                    hit_xy[i]      = (hx, hy)
                    break
                r += 0.1
            if not hit_surface[i]:
                ranges[i] = self.dvl.max_range

        return ranges, hit_surface, hit_xy

    def _simulate_altimeter(self):
        """Simulate altimeter: vertical range to seafloor directly below vehicle."""
        r = max(0.0, self._terrain_at(self.vehicle_x, self.vehicle_y) - self.vehicle_z)
        if r >= self.altimeter.max_range:
            return self.altimeter.max_range, False
        return r, True

    def _simulate_sonar(self):
        """Ray-trace sonar from vehicle nose in heading direction.

        Returns (range_from_nose, hit, hit_xy).
        """
        h = self.vehicle_heading
        cos_h, sin_h = np.cos(h), np.sin(h)
        vl = self.mapper.omap.cfg.vehicle_length
        nose_x = self.vehicle_x + vl / 2.0 * cos_h
        nose_y = self.vehicle_y + vl / 2.0 * sin_h
        n_rays    = 7
        v_angles  = np.linspace(-self.sonar.half_angle_rad,
                                 self.sonar.half_angle_rad, n_rays)
        min_range = self.sonar.max_range
        hit       = False
        hit_xy: tuple | None = None

        for v_ang in v_angles:
            r = 0.2
            while r < self.sonar.max_range:
                ds = np.cos(v_ang) * r
                dz = np.sin(v_ang) * r
                hx = nose_x + ds * cos_h
                hy = nose_y + ds * sin_h
                hz = self.vehicle_z + dz
                if hz >= self._terrain_at(hx, hy):
                    noisy_r = r + np.random.normal(0, self.sonar.noise_std)
                    if noisy_r < min_range:
                        min_range = max(0.1, noisy_r)
                        hit_xy    = (hx, hy)
                    hit = True
                    break
                r += 0.2

        return min_range, hit, hit_xy

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _cmd_depth_at_arc(self, arc: float) -> float:
        """Linearly interpolate mapper.omap.cmd_depth at along-track position arc."""
        omap = self.mapper.omap
        if omap.nx < 2:
            return self.vehicle_z
        rel = (arc - omap.grid_origin_x) / omap.cfg.dx
        ix_low = int(np.floor(rel))
        ix_high = ix_low + 1
        if ix_low < 0:
            ix_low, ix_high = 0, 1
        if ix_high >= omap.nx:
            ix_low, ix_high = omap.nx - 2, omap.nx - 1
        z_low = omap.cmd_depth[ix_low]
        z_high = omap.cmd_depth[ix_high]
        if np.isnan(z_low) and np.isnan(z_high):
            return self.vehicle_z
        if np.isnan(z_low):
            return float(z_high)
        if np.isnan(z_high):
            return float(z_low)
        t = max(0.0, min(1.0, rel - ix_low))
        return float(z_low + t * (z_high - z_low))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, dt: float):
        """Advance simulation by *dt* seconds.

        Each step: kinematics are integrated first so the vehicle pose is
        current, then sensor callbacks fire with that accurate pose, then the
        control loop ticks (at control_hz) and updates the cached velocity
        commands that will drive the next steps.

        Order:
          1. Integrate kinematics → updated (x, y, z, arc_length, t).
          2. Update heading for the new position/arc_length.
          3. Build pose from updated position and heading.
          4. Fire sensors (DVL / altimeter / sonar) with current pose.
          5. Control tick (10 Hz): pass pose to mapper, query altitude + command, update vx/vz cache.
        """
        # 1. Integrate kinematics with current cached velocity commands
        cos_h = np.cos(self.vehicle_heading)
        sin_h = np.sin(self.vehicle_heading)
        step_ds = self._ctrl_vx * dt
        if step_ds > 0:
            self.arc_length += step_ds
            self.vehicle_x  += step_ds * cos_h
            self.vehicle_y  += step_ds * sin_h
        self.vehicle_z += self._ctrl_vz * dt
        self.vehicle_z  = max(0.0, self.vehicle_z)
        self.time += dt

        # 2. Heading at updated position
        self.vehicle_heading = self.trajectory.heading_at(
            self.arc_length, self.vehicle_x, self.vehicle_y
        )

        # 3. Pose from updated state
        pose = Pose(
            north=self.vehicle_x, east=self.vehicle_y,
            depth=self.vehicle_z, heading=self.vehicle_heading,
        )

        # 4. Sensor callbacks with accurate pose
        if self.time >= self._dvl_last_t + self._dvl_period:
            dvl_ranges, dvl_hits, dvl_hit_xy = self._simulate_dvl()
            self.dvl_hit_xy = dvl_hit_xy
            self.mapper.update_sensor(SensorType.DVL, DVLMeasurement(dvl_ranges, dvl_hits), pose)
            self._dvl_last_t += self._dvl_period

        if self.time >= self._alt_last_t + self._alt_period:
            alt_range, alt_hit = self._simulate_altimeter()
            self.mapper.update_sensor(SensorType.ALTIMETER, AltimeterMeasurement(alt_range, alt_hit), pose)
            self._alt_last_t += self._alt_period

        if self.time >= self._sonar_last_t + self._sonar_period:
            sonar_range, sonar_hit, sonar_hit_xy = self._simulate_sonar()
            self.sonar_hit_xy = sonar_hit_xy
            self.mapper.update_sensor(SensorType.SONAR, SonarMeasurement(sonar_range, sonar_hit), pose)
            self._sonar_last_t += self._sonar_period

        # 5. Control tick — runs at control_hz, independent of sensor rates
        if self.time >= self._ctrl_last_t + self._ctrl_period:
            self.mapper.update_pose(pose)
            ctrl_alt = self.mapper.get_altitude()
            ctrl_cmd = self.mapper.get_control()
            c = self.mapper.omap.cfg
            self._ctrl_vx = ctrl_cmd.vx
            if ctrl_cmd.vertical_mode == 'ALT_FOLLOW':
                if not np.isnan(ctrl_alt):
                    # Positive vz = dive; positive when alt > target (need to descend)
                    self._ctrl_vz = float(np.clip(
                        ctrl_alt - ctrl_cmd.vertical_target,
                        -c.vertical_speed, c.vertical_speed,
                    ))
                else:
                    self._ctrl_vz = 0.0
            else:  # DEPTH_HOLD
                self._ctrl_vz = float(np.clip(
                    ctrl_cmd.vertical_target - self.vehicle_z,
                    -c.vertical_speed, c.vertical_speed,
                ))
            self._ctrl_last_t += self._ctrl_period

        cmd_depth = self.mapper.omap.get_commanded_depth_at_vehicle()

        self.xy_trail.append((float(self.vehicle_x), float(self.vehicle_y)))
        if len(self.xy_trail) > 2000:
            self.xy_trail = self.xy_trail[-2000:]

        terrain_z = self._terrain_at(self.vehicle_x, self.vehicle_y)
        altitude  = terrain_z - self.vehicle_z
        state = {
            'time':             self.time,
            'vehicle_x':        self.vehicle_x,
            'vehicle_y':        self.vehicle_y,
            'vehicle_z':        self.vehicle_z,
            'vehicle_heading':  self.vehicle_heading,
            'terrain_z':        terrain_z,
            'altitude':         altitude,
            'cmd_depth':        cmd_depth if not np.isnan(cmd_depth) else self.vehicle_z,
        }
        self.log.append(state)

        if self.debug:
            self._debug_check(terrain_z, altitude)

        return state

    # ------------------------------------------------------------------
    # Debug
    # ------------------------------------------------------------------

    def _debug_check(self, terrain_z: float, altitude: float):
        self._step_count += 1
        self._x_history.append(self.arc_length)
        omap = self.mapper.omap
        self._mode_history.append(omap.control_mode)
        self._mode_history_full.append(omap.control_mode)

        if self._step_count % self._PERIODIC_PRINT_STEPS == 0:
            h_deg = np.degrees(self.vehicle_heading) % 360.0
            alt_reading = self.mapper.get_altitude()
            alt_s = f"{alt_reading:5.2f}" if not np.isnan(alt_reading) else "  nan"
            print(
                f"[T={self.time:7.1f}s  X={self.vehicle_x:7.1f}m  Y={self.vehicle_y:7.1f}m  "
                f"Z={self.vehicle_z:6.2f}m  hdg={h_deg:5.1f}°  "
                f"alt={altitude:5.2f}m  terrain={terrain_z:6.2f}m  "
                f"mode={omap.control_mode:<15} sensor_alt={alt_s}m]"
            )

        if len(self._x_history) < self._STUCK_WINDOW_STEPS:
            return

        progress = self._x_history[-1] - self._x_history[0]
        all_alt_follow = (
            len(self._mode_history_full) == self._STUCK_WINDOW_STEPS
            and all(m == "ALT_FOLLOW" for m in self._mode_history_full)
        )
        is_stuck      = all_alt_follow and progress < self._STUCK_PROGRESS_MIN
        flips         = sum(1 for i in range(1, len(self._mode_history))
                            if self._mode_history[i] != self._mode_history[i - 1])
        is_oscillating = flips >= self._OSCIL_MIN_FLIPS
        too_close      = abs(self.arc_length - self._last_stuck_report_x) < 2.0

        if (is_stuck or is_oscillating) and not too_close:
            self._last_stuck_report_x = self.arc_length
            reason = []
            if is_stuck:
                reason.append(
                    f"ALT_FOLLOW <{self._STUCK_PROGRESS_MIN}m progress "
                    f"({progress:.3f}m in {self._STUCK_WINDOW_STEPS} steps)"
                )
            if is_oscillating:
                reason.append(f"mode oscillation ({flips} flips)")
            print(
                f"\n{'='*70}\n"
                f"  VEHICLE STUCK/OSCILLATING  T={self.time:.1f}s  "
                f"X={self.vehicle_x:.1f}m  Y={self.vehicle_y:.1f}m\n"
                f"  Reason: {'; '.join(reason)}\n{'='*70}"
            )

    def run(self, duration: float, dt: float = 0.1) -> list:
        """Run simulation for *duration* seconds."""
        for _ in range(int(np.ceil(duration / dt))):
            self.step(dt)
        return self.log


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='AUV Obstacle Avoidance Simulator (headless)',
    )
    parser.add_argument(
        '--terrain', default='default',
        choices=sorted(_TERRAIN_REGISTRY),
        help='Terrain type to use (default: %(default)s)',
    )
    parser.add_argument(
        '--slope-angle', type=float, default=45.0, metavar='DEG',
        help='Sawtooth slope angle in degrees, 0–90 (default: %(default)s)',
    )
    parser.add_argument(
        '--amplitude', type=float, default=10.0, metavar='M',
        help='Sawtooth tooth height in metres (default: %(default)s)',
    )
    parser.add_argument(
        '--flat-bottom', type=float, default=0.0, metavar='M',
        help='Flat seafloor distance between sawtooth teeth in metres (default: %(default)s)',
    )
    parser.add_argument(
        '--reverse', action='store_true', default=False,
        help='Reverse sawtooth direction: vertical cliff rise then gradual slope down',
    )
    parser.add_argument(
        '--duration', type=float, default=60.0, metavar='SEC',
        help='Simulation duration in seconds (default: %(default)s)',
    )
    parser.add_argument(
        '--dt', type=float, default=0.1, metavar='SEC',
        help='Simulation time step in seconds (default: %(default)s)',
    )
    args = parser.parse_args()

    kwargs = {}
    if args.terrain == 'sawtooth':
        kwargs['slope_angle_deg'] = args.slope_angle
        kwargs['amplitude'] = args.amplitude
        kwargs['flat_bottom'] = args.flat_bottom
        kwargs['reverse'] = args.reverse

    terrain_fn = make_terrain(args.terrain, **kwargs)
    print(f"Terrain: {args.terrain}"
          + (f"  slope={args.slope_angle}°  amplitude={args.amplitude}m"
             f"  flat_bottom={args.flat_bottom}m"
             + ("  reversed" if args.reverse else "")
             if args.terrain == 'sawtooth' else ""))

    sim = Simulator(terrain_fn=terrain_fn)
    log = sim.run(duration=args.duration, dt=args.dt)

    print(f"Ran {len(log)} steps over {log[-1]['time']:.1f}s")
    print(f"Final X: {log[-1]['vehicle_x']:.1f}m")
    print(f"Final Z: {log[-1]['vehicle_z']:.2f}m")
    print(f"Final altitude: {log[-1]['altitude']:.2f}m")
    print(f"Terrain at final X: {log[-1]['terrain_z']:.2f}m")
