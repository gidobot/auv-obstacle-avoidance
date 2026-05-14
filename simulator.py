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

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Callable
from occupancy_map import OccupancyMap, OccupancyMapConfig

# ---------------------------------------------------------------------------
# Terrain generators
# ---------------------------------------------------------------------------


@dataclass
class DVLConfig:
    """Nortek Nucleus 1000 DVL: 1 altimeter + 3 beams at 20° slant.

    Each beam entry is (slant_angle_deg, heading_offset_deg):
      - slant_angle: angle from vertical (0° = straight down)
      - heading_offset: angle from vehicle heading (0° = forward, 90° = starboard)
    """
    beams: list = field(default_factory=lambda: [
        ( 0.0,   0.0),   # altimeter: straight down
        (20.0,   0.0),   # beam 1: forward-down
        (20.0, 120.0),   # beam 2: right-rear-down
        (20.0, 240.0),   # beam 3: left-rear-down
    ])
    max_range: float = 20.0

    @property
    def beam_angles_rad(self) -> np.ndarray:
        """2D projection of each beam onto the vehicle's heading plane.

        Returns the effective angle from vertical (positive = forward) for use
        by the 2D occupancy map.  A beam with slant s and heading offset h
        projects to atan2(sin(s)*cos(h), cos(s)).
        """
        angles = []
        for slant_deg, h_off_deg in self.beams:
            s = np.radians(slant_deg)
            h = np.radians(h_off_deg)
            angles.append(np.arctan2(np.sin(s) * np.cos(h), np.cos(s)))
        return np.array(angles)

    @property
    def beam_directions_3d(self) -> np.ndarray:
        """Unit-vector components per beam in vehicle frame (fwd, starboard, down).

        Returns shape (n_beams, 3).  Scale each row by range to get the
        displacement from the vehicle to the terrain hit point.
        """
        dirs = []
        for slant_deg, h_off_deg in self.beams:
            s = np.radians(slant_deg)
            h = np.radians(h_off_deg)
            dirs.append((np.sin(s) * np.cos(h),   # forward component
                         np.sin(s) * np.sin(h),   # starboard component
                         np.cos(s)))              # downward component
        return np.array(dirs)


@dataclass
class SonarConfig:
    """Forward-looking sonar configuration."""
    max_range: float = 12.0     # Maximum detection range (m)
    half_angle: float = 3.0     # Half beam width (degrees)
    noise_std: float = 0.3      # Range measurement noise std (m)

    @property
    def half_angle_rad(self) -> float:
        return np.radians(self.half_angle)


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
    speed. Sensors sample the terrain to build the occupancy map.
    """

    # ------------------------------------------------------------------
    # Stuck / oscillation detection thresholds
    # ------------------------------------------------------------------
    # Slow-progress detection: only fires when the vehicle has been in
    # ALT_FOLLOW for the ENTIRE window (OBSTACLE_CLEAR and ALT_CORRECTION
    # naturally produce low forward speed during steep rises/descents).
    # The window is wide enough to span a full vertical-descent cycle.
    _STUCK_WINDOW_STEPS: int = 100      # steps (~10 s) over which progress is checked
    _STUCK_PROGRESS_MIN: float = 0.3    # min forward progress (m) across full window
    _OSCIL_WINDOW_STEPS: int = 12       # steps checked for rapid mode oscillation
    _OSCIL_MIN_FLIPS: int = 6           # min mode flips in window to flag oscillation
    _PERIODIC_PRINT_STEPS: int = 200    # print a one-liner heartbeat every N steps

    def __init__(
        self,
        omap_config: Optional[OccupancyMapConfig] = None,
        dvl_config: Optional[DVLConfig] = None,
        sonar_config: Optional[SonarConfig] = None,
        terrain_fn: Optional[Callable[[float], float]] = None,
        initial_depth: float = 0.0,
        debug: bool = True,
    ):
        self.omap = OccupancyMap(omap_config or OccupancyMapConfig())
        self.dvl = dvl_config or DVLConfig()
        self.sonar = sonar_config or SonarConfig()
        self.terrain_fn = terrain_fn or default_terrain
        self.debug = debug

        # Vehicle state
        self.vehicle_x: float = 0.0
        self.vehicle_z: float = initial_depth
        self.time: float = 0.0

        # Initialize grid centered on vehicle
        self.omap.reset(self.vehicle_x, self.vehicle_z)

        # State log for visualization
        self.log: list = []

        # Debug / stuck-detection state
        self._step_count: int = 0
        self._x_history: deque = deque(maxlen=self._STUCK_WINDOW_STEPS)
        # _mode_history_full covers the full stuck window so the ALT_FOLLOW
        # check is not fooled by an ALT_CORRECTION that finished a few steps ago.
        self._mode_history_full: deque = deque(maxlen=self._STUCK_WINDOW_STEPS)
        self._mode_history: deque = deque(maxlen=self._OSCIL_WINDOW_STEPS)
        self._last_stuck_report_x: float = -999.0  # avoid repeat reports at same x

    def _simulate_dvl(self):
        """Generate DVL observations by ray-tracing beams against terrain."""
        angles = self.dvl.beam_angles_rad
        ranges = np.zeros(len(angles))
        hit_surface = np.zeros(len(angles), dtype=bool)

        for i, ang in enumerate(angles):
            # Ray-march until we hit terrain or exceed max range
            r = 0.1
            while r < self.dvl.max_range:
                dx = np.sin(ang) * r
                dz = np.cos(ang) * r
                hit_x = self.vehicle_x + dx
                hit_z = self.vehicle_z + dz
                floor_z = self.terrain_fn(hit_x)

                if hit_z >= floor_z:
                    ranges[i] = r
                    hit_surface[i] = True
                    break
                r += 0.1

            if not hit_surface[i]:
                ranges[i] = self.dvl.max_range

        return ranges, angles, hit_surface

    def _simulate_sonar(self):
        """Generate forward sonar observation by ray-tracing."""
        # Cast rays across the sonar beam width
        n_rays = 7
        angles = np.linspace(-self.sonar.half_angle_rad,
                             self.sonar.half_angle_rad, n_rays)
        min_range = self.sonar.max_range
        hit = False

        for ang in angles:
            r = 0.2
            while r < self.sonar.max_range:
                dx = r * np.cos(ang)
                dz = r * np.sin(ang)
                hit_x = self.vehicle_x + dx
                hit_z = self.vehicle_z + dz
                floor_z = self.terrain_fn(hit_x)

                if hit_z >= floor_z:
                    # Add noise
                    noisy_r = r + np.random.normal(0, self.sonar.noise_std)
                    if noisy_r < min_range:
                        min_range = max(0.1, noisy_r)
                    hit = True
                    break
                r += 0.2

        return min_range, hit

    def _cmd_depth_at_x(self, world_x: float) -> float:
        """Linearly interpolate omap.cmd_depth at world_x."""
        omap = self.omap
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
        """
        Advance simulation by dt seconds.

        Motion model is mode-aware:
        - ALT_FOLLOW: constant forward velocity (``survey_speed``); vertical
          follows cmd_depth target, bounded by ``vertical_speed``.
        - OBSTACLE_CLEAR / ALT_CORRECTION: vertical (no forward) when the
          cmd_depth target differs from current depth, otherwise horizontal
          at survey_speed (clearance-adjustment phase).
        """
        speed = self.omap.cfg.survey_speed
        v_speed = self.omap.cfg.vertical_speed

        # --- Sensor updates ---
        dvl_ranges, dvl_angles, dvl_hits = self._simulate_dvl()
        self.omap.update_dvl_ray(
            dvl_ranges, dvl_angles,
            self.vehicle_z, self.vehicle_x,
            hit_surface=dvl_hits,
        )

        sonar_range, sonar_hit = self._simulate_sonar()
        self.omap.update_sonar(
            sonar_range, self.sonar.half_angle_rad,
            self.vehicle_z, self.vehicle_x,
            sonar_hit
        )

        # --- Build manifold and path ---
        cmd_depth = self.omap.update(self.vehicle_z)

        # --- Mode-aware motion ---
        mode = self.omap.control_mode
        max_dz = v_speed * dt

        if mode == "ALT_FOLLOW":
            forward_dx = speed * dt
            target_z = self._cmd_depth_at_x(self.vehicle_x + forward_dx)
            dz = target_z - self.vehicle_z
            if abs(dz) > max_dz:
                dz = np.sign(dz) * max_dz
            self.vehicle_z += dz
            self.vehicle_x += forward_dx
            self.omap.advance(forward_dx)
        elif mode in ("OBSTACLE_CLEAR", "ALT_CORRECTION"):
            target_z = cmd_depth if not np.isnan(cmd_depth) else self.vehicle_z
            dz = target_z - self.vehicle_z
            if abs(dz) > 0.1:
                if abs(dz) > max_dz:
                    dz = np.sign(dz) * max_dz
                self.vehicle_z += dz
            else:
                # At target — clearance-adjustment forward.
                forward_dx = speed * dt
                self.vehicle_x += forward_dx
                self.omap.advance(forward_dx)
        else:
            self.vehicle_x += speed * dt
            self.omap.advance(speed * dt)

        # Clamp vehicle depth at the water surface — vehicle cannot fly
        # above water.  Without this the OBSTACLE_CLEAR ascent can push
        # vehicle_z negative.
        if self.vehicle_z < 0.0:
            self.vehicle_z = 0.0

        self.omap.shift_depth(self.vehicle_z)
        self.time += dt

        # --- Log state ---
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

        # --- Debug output ---
        if self.debug:
            self._debug_check(terrain_z, altitude)

        return state

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    def _debug_check(self, terrain_z: float, altitude: float):
        """Check for stuck / oscillation conditions and emit diagnostics."""
        self._step_count += 1
        self._x_history.append(self.vehicle_x)
        self._mode_history.append(self.omap.control_mode)
        self._mode_history_full.append(self.omap.control_mode)

        # --- Periodic heartbeat ---
        if self._step_count % self._PERIODIC_PRINT_STEPS == 0:
            print(
                f"[T={self.time:7.1f}s  X={self.vehicle_x:7.1f}m  Z={self.vehicle_z:6.2f}m  "
                f"alt={altitude:5.2f}m  terrain={terrain_z:6.2f}m  "
                f"mode={self.omap.control_mode:<15} dvl={self.omap.dvl_altitude:5.2f}m]"
            )

        # Only run full diagnostics when we have enough history
        if len(self._x_history) < self._STUCK_WINDOW_STEPS:
            return

        progress = self._x_history[-1] - self._x_history[0]
        # Slow-progress alert only makes sense when the vehicle has been in
        # ALT_FOLLOW for the *entire* window.  OBSTACLE_CLEAR / ALT_CORRECTION
        # naturally produce low forward speed during steep rises/descents;
        # even a completed descent a few steps ago should suppress the alert.
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

        # Suppress repeat reports within 2 m of the last report
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
                f"Target alt: {self.omap.cfg.imaging_altitude:.2f}m\n"
                f"  Mode history (last {self._OSCIL_WINDOW_STEPS}): "
                f"{' '.join(m[0] for m in self._mode_history)}\n"
                + self.omap.get_debug_summary(self.vehicle_z) +
                f"\n{'='*70}"
            )

    def run(self, duration: float, dt: float = 0.1) -> list:
        """
        Run simulation for a given duration.

        Args:
            duration: Total simulation time (seconds).
            dt: Time step (seconds).

        Returns:
            List of state dicts from each step.
        """
        steps = int(np.ceil(duration / dt))
        for _ in range(steps):
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
    'default':   make_terrain_3d_default,
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
    leg_length: float = 100.0,
    spacing: float = 20.0,
    n_legs: int = 5,
    orientation_deg: float = 0.0,
    start_x: float = 0.0,
    start_y: float = 0.0,
    turn_rate: float = 0.0,
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
    same 2D obstacle avoidance system as :class:`Simulator`.  Sensor beams
    (DVL, sonar) are cast in 3D against the terrain function and projected
    onto the vehicle's along-heading axis before being fed to the occupancy
    map — so the map always represents the vertical cross-section in the
    direction of travel.

    Args:
        terrain_fn:  ``f(x, y) → depth`` callable.  Use :func:`make_terrain_3d`
            or :func:`extrude_terrain_3d` to create one.
        trajectory:  :class:`Trajectory3D` instance or ``None`` for straight.
        initial_x / initial_y / initial_heading_deg:  Starting XY position
            and heading.
        initial_depth:  Starting depth (m, positive down).  Default 0.0
            (surface) — the vehicle dives via sensor-driven path planning,
            matching how a real AUV would start a mission.
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
        terrain_fn:  Optional[Callable]            = None,
        trajectory:  Optional[Trajectory3D]        = None,
        initial_x:   float = 0.0,
        initial_y:   float = 0.0,
        initial_depth: float = 0.0,
        initial_heading_deg: float = 0.0,
        turn_voxel_mirror: bool = False,
        turn_mirror_threshold_deg: float = 30.0,
        debug: bool = True,
    ):
        self.omap  = OccupancyMap(omap_config or OccupancyMapConfig())
        self.dvl   = dvl_config   or DVLConfig()
        self.sonar = sonar_config or SonarConfig()
        self.terrain_fn = terrain_fn or make_terrain_3d('default')
        self.trajectory = trajectory or StraightTrajectory3D(initial_heading_deg)
        self.debug = debug

        # 3D vehicle state
        self.vehicle_x: float = initial_x
        self.vehicle_y: float = initial_y
        self.vehicle_z: float = initial_depth
        self.vehicle_heading: float = np.radians(initial_heading_deg)
        self.arc_length: float = 0.0   # cumulative along-track distance
        self.time: float = 0.0

        # Along-track offset so that "arc_local" fed to the occupancy map starts
        # at 0 on initialisation and increases monotonically with arc_length.
        self._arc_offset: float = 0.0

        # Turn-mirror: when the heading change accumulated within the last
        # horizon_fwd metres reaches the threshold, filled voxels behind the
        # vehicle are max-merged into the symmetric columns ahead.
        # Uses omap.turn_dh_bins — a 1-D array with the same nx and dx as the
        # occupancy grid.  Each column accumulates the heading change (rad) that
        # occurred while the vehicle was in that spatial bin.  The array slides
        # left identically to omap.grid via advance(), so the window is always
        # exactly the last horizon_fwd metres of travel.
        self.turn_voxel_mirror: bool = turn_voxel_mirror
        self._turn_mirror_threshold: float = np.radians(turn_mirror_threshold_deg)
        self._last_heading: float = np.radians(initial_heading_deg)

        self.omap.reset(0.0, initial_depth)

        # Logging / debug
        self.log: list = []
        self._step_count:       int   = 0
        self._x_history:        deque = deque(maxlen=self._STUCK_WINDOW_STEPS)
        self._mode_history:     deque = deque(maxlen=self._OSCIL_WINDOW_STEPS)
        self._mode_history_full:deque = deque(maxlen=self._STUCK_WINDOW_STEPS)
        self._last_stuck_report_x: float = -999.0

        # XY trail stored for visualizer top-down view
        self.xy_trail: list = []

        # Last beam footprints in world XY (None if beam did not hit)
        n_beams = len(self.dvl.beam_angles_rad)
        self.dvl_hit_xy:   list  = [None] * n_beams  # [(x,y) or None, ...]
        self.sonar_hit_xy: tuple | None = None        # (x, y) or None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def _arc_local(self) -> float:
        """Along-track distance from the last occupancy-grid reset."""
        return self.arc_length - self._arc_offset

    @staticmethod
    def _wrap_angle(a: float) -> float:
        return (a + np.pi) % (2.0 * np.pi) - np.pi

    def _terrain_at(self, x: float, y: float) -> float:
        try:
            return float(self.terrain_fn(x, y))
        except Exception:
            return 20.0

    def _mirror_voxels_across_vehicle(self) -> None:
        """Mirror filled voxels from behind the vehicle into the symmetric columns ahead.

        For each column d bins behind the vehicle's current grid position, the
        occupancy values are max-merged into the column d bins ahead.  Existing
        voxels in front are never decreased — this is a conservative "assume the
        terrain ahead looks like the terrain just passed" operation, intended for
        use when the vehicle turns sharply near steep walls.
        """
        ix_v = int(round((self._arc_local - self.omap.grid_origin_x) / self.omap.cfg.dx))
        nx   = self.omap.nx
        for d in range(1, nx):
            ix_behind = ix_v - d
            ix_ahead  = ix_v + d
            if ix_behind < 0 or ix_ahead >= nx:
                break
            np.maximum(self.omap.grid[:, ix_ahead],
                       self.omap.grid[:, ix_behind],
                       out=self.omap.grid[:, ix_ahead])

    # ------------------------------------------------------------------
    # Sensor simulation (3D)
    # ------------------------------------------------------------------

    def _simulate_dvl(self):
        """Simulate Nortek Nucleus 1000 DVL beams in 3D.

        Each beam is defined by (forward, starboard, down) unit-vector components
        in the vehicle frame (from ``dvl.beam_directions_3d``), rotated into world
        XY by the vehicle heading.  The 2D projected angles (``dvl.beam_angles_rad``)
        are returned alongside so the occupancy-map can still use them.

        Returns:
            ranges, angles_2d, hit_surface, hit_xy

        *hit_xy* is a list with one entry per beam; each entry is either
        ``(world_x, world_y)`` where the beam intersected the terrain, or
        ``None`` if the beam did not hit within max_range.
        """
        h = self.vehicle_heading
        cos_h, sin_h    = np.cos(h), np.sin(h)
        beam_dirs       = self.dvl.beam_directions_3d  # (n_beams, 3): fwd, stbd, down
        angles_2d       = self.dvl.beam_angles_rad
        n_beams         = len(beam_dirs)
        ranges          = np.zeros(n_beams)
        hit_surface     = np.zeros(n_beams, dtype=bool)
        hit_xy          = [None] * n_beams

        for i, (fwd, stbd, down) in enumerate(beam_dirs):
            r = 0.1
            while r < self.dvl.max_range:
                # Rotate beam direction from vehicle frame to world XY
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

        return ranges, angles_2d, hit_surface, hit_xy

    def _simulate_sonar(self):
        """Forward sonar fan in the vehicle's heading direction.

        Returns:
            min_range, hit, hit_xy

        *hit_xy* is ``(world_x, world_y)`` of the closest sonar return, or
        ``None`` if nothing was hit within max_range.
        """
        h = self.vehicle_heading
        cos_h, sin_h = np.cos(h), np.sin(h)
        n_rays    = 7
        v_angles  = np.linspace(-self.sonar.half_angle_rad,
                                 self.sonar.half_angle_rad, n_rays)
        min_range = self.sonar.max_range
        hit       = False
        hit_xy: tuple | None = None

        for v_ang in v_angles:
            r = 0.2
            while r < self.sonar.max_range:
                ds    = np.cos(v_ang) * r
                dz    = np.sin(v_ang) * r
                hx    = self.vehicle_x + ds * cos_h
                hy    = self.vehicle_y + ds * sin_h
                hz    = self.vehicle_z + dz
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
        """Linearly interpolate ``omap.cmd_depth`` at an along-track arc x.

        Returns vehicle_z (no-op fallback) if the interpolation can't be
        evaluated (NaN endpoints, out-of-grid, etc.).
        """
        omap = self.omap
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
        t = rel - ix_low
        t = max(0.0, min(1.0, t))
        return float(z_low + t * (z_high - z_low))

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, dt: float):
        """Advance simulation by *dt* seconds.

        Motion model is mode-aware:
        - ALT_FOLLOW: constant forward velocity (``survey_speed``); vertical
          follows the cmd_depth target, bounded by ``vertical_speed``.
        - OBSTACLE_CLEAR / ALT_CORRECTION: vertical motion (at
          ``vertical_speed``, zero forward) when the cmd_depth target differs
          from the vehicle's depth; otherwise horizontal motion (at
          ``survey_speed``, zero vertical) for the clearance-adjustment phase
          when the safety cap holds cmd_depth at the vehicle's depth.
        """
        speed = self.omap.cfg.survey_speed
        v_speed = self.omap.cfg.vertical_speed

        # --- Update heading from trajectory ---
        new_h = self.trajectory.heading_at(self.arc_length, self.vehicle_x, self.vehicle_y)
        dh = self._wrap_angle(new_h - self._last_heading)   # signed: +left, -right
        self.vehicle_heading = new_h
        self._last_heading   = new_h

        arc_local = self._arc_local   # along-track from last reset

        # --- Sensor updates ---
        dvl_ranges, dvl_angles, dvl_hits, dvl_hit_xy = self._simulate_dvl()
        self.dvl_hit_xy = dvl_hit_xy
        self.omap.update_dvl_ray(
            dvl_ranges, dvl_angles,
            self.vehicle_z, arc_local,
            hit_surface=dvl_hits,
        )

        sonar_range, sonar_hit, sonar_hit_xy = self._simulate_sonar()
        self.sonar_hit_xy = sonar_hit_xy
        self.omap.update_sonar(
            sonar_range, self.sonar.half_angle_rad,
            self.vehicle_z, arc_local,
            sonar_hit,
        )

        # --- Build manifold and path ---
        cmd_depth = self.omap.update(self.vehicle_z)

        # --- Mode-aware motion ---
        mode = self.omap.control_mode
        max_dz = v_speed * dt

        if mode == "ALT_FOLLOW":
            # Constant forward velocity.  Vertical tracks cmd_depth at the
            # vehicle's new arc-x, bounded by vertical_speed.
            forward_ds = speed * dt
            target_z = self._cmd_depth_at_arc(arc_local + forward_ds)
            dz = target_z - self.vehicle_z
            if abs(dz) > max_dz:
                dz = np.sign(dz) * max_dz
            self.vehicle_z += dz
            step_ds = forward_ds
        elif mode in ("OBSTACLE_CLEAR", "ALT_CORRECTION"):
            # Vertical or horizontal — never both.  cmd_depth at vehicle's
            # column tells us which:
            #   - |dz| above threshold → vertical at vertical_speed.
            #   - |dz| at threshold (cmd_depth capped at vehicle_depth by
            #     safety) → horizontal at survey_speed.
            target_z = cmd_depth if not np.isnan(cmd_depth) else self.vehicle_z
            dz = target_z - self.vehicle_z
            if abs(dz) > 0.1:
                if abs(dz) > max_dz:
                    dz = np.sign(dz) * max_dz
                self.vehicle_z += dz
                step_ds = 0.0
            else:
                step_ds = speed * dt
        else:
            step_ds = speed * dt

        if step_ds > 0:
            self.arc_length += step_ds
            self.omap.advance(step_ds)
            cos_h = np.cos(self.vehicle_heading)
            sin_h = np.sin(self.vehicle_heading)
            self.vehicle_x += step_ds * cos_h
            self.vehicle_y += step_ds * sin_h

        # --- Turn-mirror: spatial-bin sliding-window check ---
        # Accumulate SIGNED heading change into the vehicle's current column.
        # omap.advance() already slid turn_dh_bins left, so cx is always the
        # vehicle's current bin and bins 0..cx-1 hold the behind history.
        cx = self.omap.cx
        self.omap.turn_dh_bins[cx] += dh
        # Iterate backwards from cx, accumulating the signed running sum.
        # Opposite turns cancel out (e.g. 20° left + 20° right = 0°) so
        # back-and-forth oscillation never triggers.  Stop as soon as the
        # absolute running sum exceeds the threshold (short-circuit).
        if self.turn_voxel_mirror:
            horizon_bins = min(cx + 1, int(round(self.omap.cfg.horizon_fwd / self.omap.cfg.dx)))
            running = 0.0
            triggered = False
            for i in range(cx, cx - horizon_bins, -1):
                running += self.omap.turn_dh_bins[i]
                if abs(running) >= self._turn_mirror_threshold:
                    triggered = True
                    break
            if triggered:
                self._mirror_voxels_across_vehicle()
                self.omap.turn_dh_bins[:] = 0.0  # clear all bins after trigger

        # Clamp vehicle depth at the water surface — vehicle cannot fly
        # above water.
        if self.vehicle_z < 0.0:
            self.vehicle_z = 0.0

        self.omap.shift_depth(self.vehicle_z)
        self.time += dt

        # Track XY trail (keep last 2000 points)
        self.xy_trail.append((float(self.vehicle_x), float(self.vehicle_y)))
        if len(self.xy_trail) > 2000:
            self.xy_trail = self.xy_trail[-2000:]

        # --- Log state ---
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
        self._mode_history.append(self.omap.control_mode)
        self._mode_history_full.append(self.omap.control_mode)

        if self._step_count % self._PERIODIC_PRINT_STEPS == 0:
            h_deg = np.degrees(self.vehicle_heading) % 360.0
            print(
                f"[T={self.time:7.1f}s  X={self.vehicle_x:7.1f}m  Y={self.vehicle_y:7.1f}m  "
                f"Z={self.vehicle_z:6.2f}m  hdg={h_deg:5.1f}°  "
                f"alt={altitude:5.2f}m  terrain={terrain_z:6.2f}m  "
                f"mode={self.omap.control_mode:<15} dvl={self.omap.dvl_altitude:5.2f}m]"
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
