"""
AUV 2D Obstacle Avoidance - Occupancy Map, Cliff Manifold, and Path Planner

A vehicle-relative 2D occupancy grid (X forward, Z down) for seafloor imaging
AUV obstacle avoidance. The map operates like a side-scrolling window centered
on the vehicle, with a configurable forward and backward horizon.

Sensor observations from a downward-facing DVL (3 Janus beams + altimeter)
and a forward-looking sonar are projected into the 2D X-Z voxel space,
ignoring lateral (Y) offsets.

The "cliff world" manifold extracts the top surface of occupied voxels as a
stair-step polyline.  A path planner generates a commanded depth profile over
this manifold using several control modes with fixed priority:

  1) Forward obstacle clearance (highest).  While the cliff-top latch is
     active, OBSTACLE_CLEAR / OBSTACLE_HOLD dominate — ascent or forward
     clear at latch depth until ``release_x``.  Outside the latch,
     OBSTACLE_CLEAR (vehicle below commanded depth threshold) similarly
     takes precedence — no tail-only override.

  2) Tail clearance vs altitude following / correction.  If the tail
     safety band fires (see ``_safety_tail_blocked``) and the vehicle would
     otherwise be ALT_FOLLOW or ALT_CORRECTION, switch to TAIL_CLEAR constant-
     depth forward flight until the tail clears.

  3) Else ALT_FOLLOW (terrain-following / imaging_altitude),
     or ALT_CORRECTION (descend in place when altitude is high).

Individual modes:

  Mode: ALT_FOLLOW  (default when forward + tail constraints satisfied)
    Track imaging_altitude above the raw manifold (terrain-following).

  Mode: OBSTACLE_CLEAR  (forward obstacle or commanded climb, vehicle deep)
    Vehicle ascends in place (vx=0) toward the latch target depth, or toward
    ``cmd_depth[cx]`` when no latch applies.

  Mode: OBSTACLE_HOLD  (latch active, vehicle at or above target depth)
    Vehicle flies forward at survey_speed holding the latch target depth
    (DEPTH_HOLD).  Active until the vehicle passes release_x (target_x +
    cliff_standoff + vehicle_length m).  The latch can only be updated
    to a shallower target — never overridden or released early.

  Mode: ALT_CORRECTION  (altitude diverges above imaging altitude target)
    Triggered when vehicle depth is shallower than ``cmd_depth[cx]`` by more
    than ``altitude_overshoot_threshold_m``.  Vehicle stops forward motion
    and descends in place toward imaging altitude unless TAIL_CLEAR applies.

  Mode: TAIL_CLEAR  (tail clearance — overrides ALT_FOLLOW / ALT_CORRECTION)
    Terrain within ``safety_below_m`` below the vehicle is detected in the
    tail window (vehicle centre back to ``safety_standoff_m`` behind the
    tail).  Vehicle drives forward at survey_speed at current depth
    (DEPTH_HOLD) until ``_safety_tail_blocked`` clears, without diving for
    imaging altitude.

Coordinate conventions:
    X: forward (positive ahead of vehicle)
    Z: depth (positive downward)
    Y: lateral (ignored in 2D projection)

Usage:
    from occupancy_map import OccupancyMap

    omap = OccupancyMap()
    omap.update_dvl(dvl_ranges, dvl_beam_angles, vehicle_depth)
    omap.update_sonar(sonar_range, vehicle_depth)
    omap.advance(dx)  # call as vehicle moves forward
    manifold = omap.get_cliff_manifold()
    path = omap.get_commanded_depth_path()
"""

import threading
import numpy as np
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Union


@dataclass
class DVLConfig:
    """DVL beam configuration.

    Each beam entry is (slant_angle_deg, heading_offset_deg):
      - slant_angle: angle from vertical (0 = straight down)
      - heading_offset: angle from vehicle forward (0 = forward, 90 = starboard)

    Default: Nortek Nucleus 1000 — 1 altimeter + 3 Janus beams at 20° slant.
    """
    beams: list = field(default_factory=lambda: [
        (20.0,   0.0),   # beam 1: forward-down
        (20.0, 120.0),   # beam 2: right-rear-down
        (20.0, 240.0),   # beam 3: left-rear-down
    ])
    # Simulator-only: the ray-caster needs a cutoff at which to stop marching
    # and report a no-return.  A real DVL applies its own range limit and
    # reports per-beam validity, so vehicle-side adapters must gate on those
    # flags rather than on a configured maximum.
    max_range: float = 50.0

    @property
    def beam_angles_rad(self) -> np.ndarray:
        """2D projected angle from vertical (positive = forward) for each beam.

        A beam with slant s and heading offset h projects to
        atan2(sin(s)*cos(h), cos(s)).
        """
        angles = []
        for slant_deg, h_off_deg in self.beams:
            s = np.radians(slant_deg)
            h = np.radians(h_off_deg)
            angles.append(np.arctan2(np.sin(s) * np.cos(h), np.cos(s)))
        return np.array(angles)

    @property
    def beam_directions_3d(self) -> np.ndarray:
        """Unit vectors per beam in vehicle frame (forward, starboard, down).

        Shape (n_beams, 3). Scale by range to get displacement to terrain hit.
        """
        dirs = []
        for slant_deg, h_off_deg in self.beams:
            s = np.radians(slant_deg)
            h = np.radians(h_off_deg)
            dirs.append((np.sin(s) * np.cos(h),
                         np.sin(s) * np.sin(h),
                         np.cos(s)))
        return np.array(dirs)

    @property
    def beam_can_clear(self) -> np.ndarray:
        """Boolean array: True for beams whose 3-D direction lies in the
        vehicle X-Z plane (no lateral/starboard displacement).

        Only axis-aligned beams may clear voxels in the 2-D occupancy grid.
        Sideways beams travel through different 3-D voxels than their 2-D
        projection implies, so clearing along their projected path would
        incorrectly free voxels the beam never actually passed through.
        """
        dirs = self.beam_directions_3d
        return np.abs(dirs[:, 1]) < 1e-9


@dataclass
class SonarConfig:
    """Forward-looking sonar configuration."""
    max_range: float = 12.0
    half_angle: float = 3.0    # half beam width (degrees)
    noise_std: float = 0.3     # range measurement noise std (m)

    @property
    def half_angle_rad(self) -> float:
        return np.radians(self.half_angle)


@dataclass
class AltimeterConfig:
    """Downward-looking altimeter configuration."""
    max_range: float = 100.0


@dataclass
class Pose:
    """Vehicle pose in NED frame.

    heading: compass radians, 0 = North, clockwise positive.
    depth: positive downward (m).
    """
    north: float
    east: float
    depth: float
    heading: float


class SensorType(Enum):
    DVL = "dvl"
    ALTIMETER = "altimeter"
    SONAR = "sonar"


@dataclass
class DVLMeasurement:
    """Range measurement from a DVL instrument."""
    ranges: np.ndarray      # range per beam (m), shape (n_beams,)
    hit_surface: np.ndarray # bool per beam, shape (n_beams,)


@dataclass
class AltimeterMeasurement:
    """Single-beam downward-looking altimeter measurement."""
    range_m: float
    hit: bool = True


@dataclass
class SonarMeasurement:
    """Forward sonar closest-return measurement."""
    range_m: float   # range from sonar face (nose of vehicle)
    hit: bool        # True if a return was detected


@dataclass
class ControlCommand:
    """Obstacle avoidance command returned by ObstacleMapper.get_control().

    vx:              Desired forward speed (m/s).  Zero during a vertical
                     transit in DEPTH_HOLD mode; survey_speed otherwise.
    vertical_mode:   'ALT_FOLLOW' — external controller should maintain
                     *vertical_target* metres altitude above the seafloor.
                     'DEPTH_HOLD' — external controller should hold the vehicle
                     at exactly *vertical_target* metres depth.
    vertical_target: ALT_FOLLOW: desired altitude above seafloor (m).
                     DEPTH_HOLD: desired vehicle depth (m, positive = down).
    """
    vx: float
    vertical_mode: str    # 'ALT_FOLLOW' | 'DEPTH_HOLD'
    vertical_target: float


@dataclass
class OccupancyMapConfig:
    """Configuration parameters for the occupancy map and path planner."""

    # Voxel grid dimensions
    dx: float = 0.5          # X bin size (m) - forward axis
    dz: float = 0.25         # Z bin size (m) - depth axis, finer for altitude
    horizon_fwd: float = 15.0   # Forward look-ahead distance (m)
    horizon_back: float = 15.0  # Backward look-behind distance (m)
    z_half_range: float = 20.0  # Half-height of the sliding depth window (m).
                                 # The grid spans [vehicle_z - z_half_range,
                                 # vehicle_z + z_half_range] and shifts with
                                 # the vehicle to keep it centred.

    # Vehicle parameters
    vehicle_length: float = 2.0   # Vehicle length (m) for tail clearance
    imaging_altitude: float = 2.0  # Target altitude above seafloor (m).
                                    # Also defines the depth band used by the
                                    # tail-clearance check: structure within
                                    # [v_z, v_z + imaging_altitude) below the
                                    # vehicle counts as a tail threat.
    survey_speed: float = 0.5     # Forward survey speed (m/s)
    vertical_speed: float = 0.5   # Vertical transit speed (m/s)

    # Path planning
    cliff_standoff: float = 2.0   # Clearance distance (m): obstacle/drop trigger
                                   # window ahead, AND minimum horizontal clearance
                                   # behind the vehicle from manifold geometry in
                                   # the tail depth band before descent is allowed.
    obstacle_threshold: float = 1.0  # Minimum vertical rise/drop (m) within
                                      # cliff_standoff ahead of the vehicle to
                                      # trigger obstacle clearance or cliff descent
                                      # hold.  Gradual slopes below this threshold
                                      # are tracked in altitude-following mode.

    altitude_overshoot_threshold_m: float = 1.0  # If the vehicle's altitude
                                                  # exceeds imaging_altitude by
                                                  # more than this, planner
                                                  # switches to ALT_CORRECTION:
                                                  # vehicle stops forward motion
                                                  # and descends in place toward
                                                  # imaging altitude.  Catches
                                                  # the case where the slope is
                                                  # too steep for ALT_FOLLOW's
                                                  # vertical_speed-bounded
                                                  # descent to keep up.

    # Hysteresis on top of altitude_overshoot_threshold_m when choosing
    # OBSTACLE_CLEAR vs ALT_FOLLOW vs ALT_CORRECTION.  Once in an in-place
    # vertical transect mode, retain it until commanded depth agrees with the
    # vehicle depth within roughly (threshold - hysteresis); damps chatter when
    # noisy manifold/DVL jitter steps ``cmd_depth[cx]`` across the threshold
    # (common on procedural / rugose reef terrain).
    altitude_overshoot_hysteresis_m: float = 0.5

    # Stale-observation heading gate.  Any occupied voxel whose stored
    # observation heading differs from the current vehicle heading by more
    # than this threshold is reset to prior probability.  Prevents filled
    # voxels from a previous heading from triggering obstacle-avoidance on
    # a new heading where that space is actually clear.
    stale_heading_threshold_deg: float = 45.0

    # Safety clearance check (always active).  Fires when any observed manifold
    # column is found within ``safety_below_m`` below the vehicle AND within
    # the horizontal zone from the vehicle nose (vehicle_x + vehicle_length/2)
    # back ``safety_standoff_m`` metres behind the vehicle centre.  Used both
    # to cap cmd_depth (prevent diving when terrain is close below the body)
    # and by get_control() to decide whether descent-in-place is safe.
    safety_standoff_m: float = 2.0    # Distance behind vehicle centre (m)
    safety_below_m: float = 1.0       # Depth band below vehicle (m)

    # Occupancy probability parameters
    prior: float = 0.5            # Prior occupancy probability
    occ_thresh: float = 0.62      # Threshold to consider a voxel occupied

    # DVL observation model
    dvl_hit_prob: float = 0.5     # P(occupied | hit) increment
    dvl_miss_prob: float = 0.3    # P(free | miss) decrement
    dvl_max_occ: float = 0.98     # Max occupancy from DVL
    dvl_min_occ: float = 0.02     # Min occupancy from DVL

    # Altimeter observation model
    altimeter_hit_prob: float = 0.5   # P(occupied | hit) increment
    altimeter_miss_prob: float = 0.3  # P(free | miss) decrement
    altimeter_max_occ: float = 0.98   # Max occupancy from altimeter
    altimeter_min_occ: float = 0.02   # Min occupancy from altimeter

    # Forward sonar observation model
    sonar_hit_prob: float = 0.3   # P(occupied | hit) increment
    sonar_miss_prob: float = 0.2  # P(free | miss) decrement
    sonar_max_occ: float = 0.98   # Max occupancy from sonar
    sonar_min_occ: float = 0.02   # Min occupancy from sonar
    sonar_min_depth_m: float = 1.0  # Ignore sonar returns when vehicle depth < this (surface reflection rejection)


class OccupancyMap:
    """
    2D vehicle-relative occupancy grid for AUV obstacle avoidance.

    The grid is indexed as grid[iz, ix] where:
        ix: column index (X forward axis), 0 = back of look-behind window
        iz: row index (Z depth axis), 0 = shallowest depth

    The vehicle is always at column index cx (center of grid).
    As the vehicle advances, columns shift left and new unexplored
    columns enter from the right.

    Control operates in three modes (see module docstring) derived freshly
    each cycle from the raw occupancy manifold.  No high-water accumulation
    is used; the raw manifold drives all decisions.
    """

    def __init__(self, config: Optional[OccupancyMapConfig] = None):
        self.cfg = config or OccupancyMapConfig()
        c = self.cfg

        # Grid dimensions
        self.nx = int(np.ceil((c.horizon_fwd + c.horizon_back) / c.dx))
        self.nz = int(np.ceil(2.0 * c.z_half_range / c.dz))
        self.cx = int(np.floor(c.horizon_back / c.dx))  # Vehicle column index

        # Occupancy grid: probability of occupancy [0, 1]
        self.grid = np.full((self.nz, self.nx), c.prior, dtype=np.float64)

        # Per-voxel observation heading (radians).  Stores the vehicle heading
        # at the time a voxel was last marked occupied.  NaN for voxels with no
        # occupied observation (free or unobserved).  Slides with the grid via
        # advance() / shift_depth().  Used by clear_stale_voxels() to invalidate
        # occupied voxels that are inconsistent with the current vehicle heading.
        self.voxel_heading: np.ndarray = np.full(
            (self.nz, self.nx), np.nan, dtype=np.float32
        )

        # Grid origin in world X coordinates
        self.grid_origin_x: float = 0.0  # World X of column 0

        # Grid origin in world Z (depth) — row 0 corresponds to this depth.
        # Shifts with the vehicle so the vehicle stays centred in the window.
        self.grid_origin_z: float = -(self.nz // 2) * c.dz  # centred at depth 0 initially

        # Shift accumulator for sub-bin advances
        self._shift_accum: float = 0.0
        self._shift_accum_z: float = 0.0

        # Derived outputs (updated by build methods).  manifold_z is initialised
        # to the grid bottom depth so a manifold always exists at every column
        # ("we don't know where the terrain is, so assume it's at the depth
        # limit of the occupancy map").  build_cliff_manifold fills it each
        # cycle with:
        #   - direct observations (any column where the occupancy grid has a
        #     voxel above occ_thresh),
        #   - grid-bottom default for unobserved columns BEHIND the vehicle
        #     (no backward forward-extension — a single backward beam hit
        #     never propagates into adjacent never-observed columns),
        #   - forward-extension of the last observed column AHEAD of the
        #     vehicle into still-unobserved columns further ahead,
        #   - grid-bottom default for unobserved columns at and ahead of the
        #     vehicle before any ahead observation has been made.
        self.manifold_iz: np.ndarray = np.full(self.nx, self.nz - 1, dtype=np.int32)
        self.manifold_z: np.ndarray = np.full(
            self.nx, self.grid_origin_z + (self.nz - 1) * c.dz, dtype=np.float64
        )
        # Parallel "is this column based on a real observation" flag.  False
        # where manifold_z is the grid-bottom default — those columns are
        # truly unknown and the planner's obstacle-avoidance checks (climb,
        # cliff drop, tail clearance) must skip transitions involving them
        # to avoid false triggers from observed→default discontinuities.
        # True for direct observations AND forward-extended ahead columns
        # (both reflect real terrain knowledge).
        self.manifold_observed: np.ndarray = np.zeros(self.nx, dtype=bool)
        self.cmd_depth: np.ndarray = np.full(self.nx, np.nan, dtype=np.float64)
        self.path_waypoints: list = []

        # Grid origin that was in effect when manifold_z was last computed.
        # Kept separately from grid_origin_x so the visualizer can map
        # manifold_z[i] → world_x = manifold_grid_origin_x + i * dx even
        # after advance() has shifted grid_origin_x forward by one column.
        self.manifold_grid_origin_x: float = 0.0

        # Direct DVL altitude: minimum vertical-component range across surface-
        # hitting beams, updated each cycle by update_dvl_ray.  ``NaN`` when no
        # beam had a valid bottom return this cycle (do not reuse stale values).
        # Used as the altitude-following reference at the vehicle's current column
        # instead of the occupancy-filtered manifold, giving faster terrain response.
        self.dvl_altitude: float = np.nan

        # Current control mode at vehicle position (for display / logging).
        self.control_mode: str = "ALT_FOLLOW"

        # Cliff-top crossing latch.  Set when the vehicle crosses over the
        # highest detected voxel within cliff_standoff ahead during a climb,
        # cleared after the vehicle has flown
        # (cliff_standoff + vehicle_length) past the peak at constant
        # depth (tail-clearance distance), or when the vehicle turns away.
        self._cliff_top_committed: bool = False
        self._cliff_top_target_z: float = np.nan  # ratchets shallower only
        self._cliff_top_target_x: float = np.nan  # ratchets forward only
        self._cliff_top_release_x: float = np.nan
        self._cliff_top_commit_heading: float = np.nan  # heading at commit

        self._last_vehicle_heading: float = np.nan  # set by update()


    def reset(self, vehicle_world_x: float = 0.0, vehicle_depth: float = 0.0):
        """Reset the grid to prior probability and re-center on vehicle."""
        c = self.cfg
        self.grid[:, :] = c.prior
        self.voxel_heading[:, :] = np.nan
        self.grid_origin_x = vehicle_world_x - self.cx * c.dx
        self.grid_origin_z = vehicle_depth - (self.nz // 2) * c.dz
        self.manifold_grid_origin_x = self.grid_origin_x
        self._shift_accum   = 0.0
        self._shift_accum_z = 0.0
        self.manifold_iz[:] = self.nz - 1
        self.manifold_z[:] = self.grid_origin_z + (self.nz - 1) * c.dz
        self.manifold_observed[:] = False
        self.cmd_depth[:] = vehicle_depth
        self.path_waypoints = []
        self.dvl_altitude = np.nan
        self.control_mode = "ALT_FOLLOW"
        self._cliff_top_committed = False
        self._cliff_top_target_z = np.nan
        self._cliff_top_target_x = np.nan
        self._cliff_top_release_x = np.nan
        self._cliff_top_commit_heading = np.nan
        self._last_vehicle_heading = np.nan

    # -------------------------------------------------------------------------
    # Coordinate transforms
    # -------------------------------------------------------------------------

    def world_to_grid(self, world_x: float, world_z: float) -> tuple:
        """Convert world (X, Z) to grid indices (ix, iz)."""
        ix = round((world_x - self.grid_origin_x) / self.cfg.dx)
        iz = round((world_z - self.grid_origin_z) / self.cfg.dz)
        return ix, iz

    def grid_to_world_x(self, ix: int) -> float:
        """Convert grid column index to world X coordinate."""
        return self.grid_origin_x + ix * self.cfg.dx

    def grid_to_world_z(self, iz: int) -> float:
        """Convert grid row index to world Z (depth) coordinate."""
        return self.grid_origin_z + iz * self.cfg.dz

    def _in_bounds(self, ix: int, iz: int) -> bool:
        return 0 <= ix < self.nx and 0 <= iz < self.nz

    # -------------------------------------------------------------------------
    # Grid management
    # -------------------------------------------------------------------------

    def advance(self, dx: float):
        """
        Advance the vehicle forward by dx meters in world X.
        Shifts the grid left by the appropriate number of columns,
        filling new columns with the prior probability.

        Call this every control cycle with the forward distance traveled.
        """
        self._shift_accum += dx
        cols = int(np.floor(self._shift_accum / self.cfg.dx))
        if cols <= 0:
            return

        self._shift_accum -= cols * self.cfg.dx
        self.grid_origin_x += cols * self.cfg.dx

        if cols >= self.nx:
            self.grid[:, :] = self.cfg.prior
            self.voxel_heading[:, :] = np.nan
        else:
            self.grid[:, :-cols] = self.grid[:, cols:]
            self.grid[:, -cols:] = self.cfg.prior
            self.voxel_heading[:, :-cols] = self.voxel_heading[:, cols:]
            self.voxel_heading[:, -cols:] = np.nan

    def shift_depth(self, vehicle_z: float) -> None:
        """Shift the Z window so the vehicle stays centred in the depth grid.

        Analogous to :meth:`advance` for the X axis.  Call every control cycle
        with the vehicle's current world depth so the grid tracks the vehicle
        vertically, keeping memory use constant regardless of how deep the
        vehicle descends.
        """
        center_iz = self.nz // 2
        vehicle_iz = (vehicle_z - self.grid_origin_z) / self.cfg.dz
        # Accumulate sub-bin motion to avoid premature shifts
        self._shift_accum_z += vehicle_iz - center_iz
        rows = int(np.floor(abs(self._shift_accum_z)))
        if rows == 0:
            return
        rows = rows if self._shift_accum_z >= 0 else -rows
        self._shift_accum_z -= rows

        self.grid_origin_z += rows * self.cfg.dz

        if rows > 0:
            # Vehicle moved deeper — drop shallow rows, expose new deep rows
            if rows >= self.nz:
                self.grid[:, :] = self.cfg.prior
                self.voxel_heading[:, :] = np.nan
            else:
                self.grid[:-rows, :] = self.grid[rows:, :]
                self.grid[-rows:, :] = self.cfg.prior
                self.voxel_heading[:-rows, :] = self.voxel_heading[rows:, :]
                self.voxel_heading[-rows:, :] = np.nan
        else:
            # Vehicle moved shallower — drop deep rows, expose new shallow rows
            r = -rows
            if r >= self.nz:
                self.grid[:, :] = self.cfg.prior
                self.voxel_heading[:, :] = np.nan
            else:
                self.grid[r:, :] = self.grid[:-r, :]
                self.grid[:r, :] = self.cfg.prior
                self.voxel_heading[r:, :] = self.voxel_heading[:-r, :]
                self.voxel_heading[:r, :] = np.nan

    # -------------------------------------------------------------------------
    # Sensor observation updates
    # -------------------------------------------------------------------------

    def update_dvl(
        self,
        ranges: np.ndarray,
        beam_angles: np.ndarray,
        vehicle_depth: float,
        vehicle_world_x: float,
        hit_surface: np.ndarray,
    ):
        """
        Update occupancy from DVL beam observations.

        The DVL beams are projected into 2D X-Z space. The Y lateral offset
        is ignored — if two beams land in the same (ix, iz) bin but different
        Y bins, they observe the same 2D cell.

        Args:
            ranges: Array of range measurements per beam (m). Shape (n_beams,).
            beam_angles: Array of beam angles from vertical (rad).
                         Positive = forward, 0 = straight down. Shape (n_beams,).
            vehicle_depth: Current vehicle depth (m).
            vehicle_world_x: Current vehicle world X position (m).
            hit_surface: Boolean array, True if beam hit the seafloor. Shape (n_beams,).
        """
        c = self.cfg
        for i in range(len(ranges)):
            r = ranges[i]
            ang = beam_angles[i]

            dx = np.sin(ang) * r
            dz = np.cos(ang) * r
            hit_world_x = vehicle_world_x + dx
            hit_world_z = vehicle_depth + dz

            ix, iz = self.world_to_grid(hit_world_x, hit_world_z)
            if not self._in_bounds(ix, iz):
                continue

            if hit_surface[i]:
                self.grid[iz, ix] = min(c.dvl_max_occ,
                                        self.grid[iz, ix] + c.dvl_hit_prob)
            else:
                self.grid[iz, ix] = max(c.dvl_min_occ,
                                        self.grid[iz, ix] - c.dvl_miss_prob)

    def update_dvl_ray(
        self,
        ranges: np.ndarray,
        beam_angles: np.ndarray,
        vehicle_depth: float,
        vehicle_world_x: float,
        hit_surface: Optional[np.ndarray] = None,
        range_step: float = 0.15,
        vehicle_heading: float = np.nan,
        can_clear: Optional[np.ndarray] = None,
    ):
        """
        Update occupancy by ray-marching each DVL beam, and track direct altitude.

        Marks cells along the beam as free, and the endpoint cell as occupied.
        This provides more information per observation than endpoint-only updates.

        Also computes self.dvl_altitude — the minimum vertical-axis range
        component across all beams that actually hit the seafloor:

            dvl_altitude = min( range_i * cos(angle_i) )   for hit beams

        If no beam records a valid bottom return, ``self.dvl_altitude`` is set
        to ``NaN`` so planners do not reuse a stale altitude from a previous
        cycle.  Beam validity comes entirely from ``hit_surface`` — the caller's
        adapter is responsible for translating its sensor's own no-return
        convention into that flag.

        This is the shortest terrain clearance observed by any beam, measured
        along the depth axis.  The altimeter (angle=0) contributes directly;
        Janus beams at ±25° contribute via their cos(25°) ≈ 0.906 factor.
        On rising terrain the forward Janus beam may detect shallower ground
        before the altimeter, giving earlier terrain-following response.

        dvl_altitude is updated only when hit_surface is provided.  Pass
        hit_surface=None to perform occupancy updates without touching the
        altitude estimate (e.g. when hit flags are unavailable).

        Args:
            ranges: Range measurements per beam (m). Shape (n_beams,).
            beam_angles: Beam angles from vertical (rad). Shape (n_beams,).
            vehicle_depth: Current vehicle depth (m).
            vehicle_world_x: Current vehicle world X position (m).
            hit_surface: Boolean array, True if beam hit the seafloor.
                         Shape (n_beams,).  If None, altitude is not updated.
            range_step: Step size for ray marching (m).
        """
        c = self.cfg
        for i in range(len(ranges)):
            r_max = ranges[i]
            if not np.isfinite(r_max):
                continue
            ang = beam_angles[i]
            is_hit = (hit_surface is None) or bool(hit_surface[i])
            allow_clear = (can_clear is None) or bool(can_clear[i])

            # Ray-march free cells — axis-aligned beams only.
            # Lateral beams travel through different 3-D voxels than their 2-D
            # projection implies; clearing along their projected path would
            # incorrectly free voxels the beam never actually passed through.
            if allow_clear:
                r = range_step
                while r < r_max - range_step:
                    dx = np.sin(ang) * r
                    dz = np.cos(ang) * r
                    ix, iz = self.world_to_grid(vehicle_world_x + dx,
                                                vehicle_depth + dz)
                    if self._in_bounds(ix, iz):
                        self.grid[iz, ix] = max(c.dvl_min_occ,
                                                self.grid[iz, ix] - c.dvl_miss_prob)
                        if self.grid[iz, ix] <= c.occ_thresh:
                            self.voxel_heading[iz, ix] = vehicle_heading
                    r += range_step

            # Endpoint: always mark hit occupied; only clear on miss if axis-aligned.
            dx = np.sin(ang) * r_max
            dz = np.cos(ang) * r_max
            ix, iz = self.world_to_grid(vehicle_world_x + dx,
                                        vehicle_depth + dz)
            if self._in_bounds(ix, iz):
                if is_hit:
                    self.grid[iz, ix] = min(c.dvl_max_occ,
                                            self.grid[iz, ix] + c.dvl_hit_prob)
                    self.voxel_heading[iz, ix] = vehicle_heading
                elif allow_clear:
                    self.grid[iz, ix] = max(c.dvl_min_occ,
                                            self.grid[iz, ix] - c.dvl_miss_prob)
                    self.voxel_heading[iz, ix] = vehicle_heading

        # Update direct altitude estimate from surface-hitting beams.
        if hit_surface is not None:
            min_vert = np.inf
            for i in range(len(ranges)):
                if hit_surface[i]:
                    # Vertical component of this beam's range (depth axis)
                    vert = ranges[i] * np.cos(beam_angles[i])
                    if vert < min_vert:
                        min_vert = vert
            if min_vert < np.inf:
                self.dvl_altitude = min_vert
            else:
                self.dvl_altitude = np.nan

    def update_altimeter_ray(
        self,
        range_m: float,
        vehicle_depth: float,
        vehicle_world_x: float,
        hit: bool,
        range_step: float = 0.15,
        vehicle_heading: float = np.nan,
    ):
        """Update occupancy from a straight-down altimeter beam.

        Ray-marches vertically from vehicle depth to range_m, marking cells
        free along the path and occupied (or free on miss) at the endpoint.

        Args:
            range_m:          Measured range to seafloor (m).
            vehicle_depth:    Current vehicle depth (m).
            vehicle_world_x:  Vehicle X position in world frame (m).
            hit:              True if the beam returned a valid seafloor return.
            range_step:       Ray-march step size (m).
            vehicle_heading:  Vehicle heading (rad) for voxel metadata.
        """
        if not np.isfinite(range_m):
            return
        c = self.cfg
        r = range_step
        while r < range_m - range_step:
            ix, iz = self.world_to_grid(vehicle_world_x, vehicle_depth + r)
            if self._in_bounds(ix, iz):
                self.grid[iz, ix] = max(c.altimeter_min_occ,
                                        self.grid[iz, ix] - c.altimeter_miss_prob)
                if self.grid[iz, ix] <= c.occ_thresh:
                    self.voxel_heading[iz, ix] = vehicle_heading
            r += range_step

        ix, iz = self.world_to_grid(vehicle_world_x, vehicle_depth + range_m)
        if self._in_bounds(ix, iz):
            if hit:
                self.grid[iz, ix] = min(c.altimeter_max_occ,
                                        self.grid[iz, ix] + c.altimeter_hit_prob)
                self.voxel_heading[iz, ix] = vehicle_heading
            else:
                self.grid[iz, ix] = max(c.altimeter_min_occ,
                                        self.grid[iz, ix] - c.altimeter_miss_prob)
                self.voxel_heading[iz, ix] = vehicle_heading

    def update_sonar(
        self,
        sonar_range: float,
        sonar_half_angle: float,
        vehicle_depth: float,
        vehicle_world_x: float,
        hit_obstacle: bool,
        range_step: float = 0.2,
        angle_steps: int = 7,
        vehicle_heading: float = np.nan,
    ):
        """
        Update occupancy from forward-looking sonar observation.

        The sonar cone is projected into the X-Z plane. Uses lower confidence
        updates than DVL to avoid false positive obstacle detections.

        Args:
            sonar_range: Measured range (m), or max range if no return.
            sonar_half_angle: Half-angle of sonar beam (rad).
            vehicle_depth: Current vehicle depth (m).
            vehicle_world_x: Current vehicle world X position (m).
            hit_obstacle: True if sonar detected a return.
            range_step: Step size for ray marching (m).
            angle_steps: Number of angular samples across the beam.
        """
        if not np.isfinite(sonar_range):
            return
        c = self.cfg
        if vehicle_depth < c.sonar_min_depth_m:
            return
        angles = np.linspace(-sonar_half_angle, sonar_half_angle, angle_steps)

        for ang in angles:
            r = range_step
            while r < sonar_range - range_step:
                dx = r * np.cos(ang)
                dz = r * np.sin(ang)
                ix, iz = self.world_to_grid(vehicle_world_x + dx,
                                            vehicle_depth + dz)
                if self._in_bounds(ix, iz):
                    self.grid[iz, ix] = max(c.sonar_min_occ,
                                            self.grid[iz, ix] - c.sonar_miss_prob)
                    if self.grid[iz, ix] <= c.occ_thresh:
                        self.voxel_heading[iz, ix] = vehicle_heading
                r += range_step

            dx = sonar_range * np.cos(ang)
            dz = sonar_range * np.sin(ang)
            ix, iz = self.world_to_grid(vehicle_world_x + dx,
                                        vehicle_depth + dz)
            if self._in_bounds(ix, iz):
                if hit_obstacle:
                    self.grid[iz, ix] = min(c.sonar_max_occ,
                                            self.grid[iz, ix] + c.sonar_hit_prob)
                    self.voxel_heading[iz, ix] = vehicle_heading
                else:
                    self.grid[iz, ix] = max(c.sonar_min_occ,
                                            self.grid[iz, ix] - c.sonar_miss_prob)
                    self.voxel_heading[iz, ix] = vehicle_heading

    # -------------------------------------------------------------------------
    # Cliff manifold extraction
    # -------------------------------------------------------------------------

    def build_cliff_manifold(self):
        """
        Extract the cliff manifold from the occupancy grid.

        For each X column, finds the shallowest (topmost) occupied voxel.  The
        manifold runs along the top of occupied space as a stair-step polyline
        — it always goes over obstacles, never underneath.

        Every column always has a manifold value.  Unobserved columns default
        to the grid bottom depth ("we don't know — assume terrain is at the
        depth limit of the occupancy map").  Direct observations override the
        default where present.  AHEAD of the vehicle, forward-extension fills
        still-unobserved columns past the last ahead observation with the
        last observed depth (flat extension assumption).  BEHIND the vehicle
        there is NO backward forward-extension: a single backward beam hit
        does not propagate into adjacent never-observed columns.

        Updates self.manifold_iz and self.manifold_z in place.
        """
        c = self.cfg
        # Snapshot the origin now so the visualizer can correctly map
        # manifold_z[i] → world_x after advance() has shifted grid_origin_x.
        self.manifold_grid_origin_x = self.grid_origin_x
        bottom_iz = self.nz - 1
        bottom_z = self.grid_to_world_z(bottom_iz)

        # Pass 1: shallowest occupied voxel per column (-1 if none).
        observed_iz = np.full(self.nx, -1, dtype=np.int32)
        for ix in range(self.nx):
            for iz in range(self.nz):
                if self.grid[iz, ix] > c.occ_thresh:
                    observed_iz[ix] = iz
                    break

        # Pass 2 — behind the vehicle: observation OR grid-bottom default.
        for ix in range(self.cx):
            if observed_iz[ix] >= 0:
                self.manifold_iz[ix] = observed_iz[ix]
                self.manifold_z[ix] = self.grid_to_world_z(observed_iz[ix])
                self.manifold_observed[ix] = True
            else:
                self.manifold_iz[ix] = bottom_iz
                self.manifold_z[ix] = bottom_z
                self.manifold_observed[ix] = False

        # Pass 3 — at and ahead of the vehicle: observation, else forward-
        # extend from the last ahead observation, else grid-bottom default.
        last_iz = -1
        for ix in range(self.cx, self.nx):
            if observed_iz[ix] >= 0:
                self.manifold_iz[ix] = observed_iz[ix]
                self.manifold_z[ix] = self.grid_to_world_z(observed_iz[ix])
                last_iz = observed_iz[ix]
                self.manifold_observed[ix] = True
            elif last_iz >= 0:
                self.manifold_iz[ix] = last_iz
                self.manifold_z[ix] = self.grid_to_world_z(last_iz)
                self.manifold_observed[ix] = True
            else:
                self.manifold_iz[ix] = bottom_iz
                self.manifold_z[ix] = bottom_z
                self.manifold_observed[ix] = False

    def get_cliff_manifold(self) -> tuple:
        """
        Returns the cliff manifold as world coordinates.

        Returns:
            (manifold_world_x, manifold_world_z): Arrays of world X and Z
                coordinates for each grid column.  Always defined — unobserved
                columns return the grid bottom depth.
        """
        world_x = np.array([self.grid_to_world_x(ix) for ix in range(self.nx)])
        return world_x, self.manifold_z.copy()

    # -------------------------------------------------------------------------
    # Path planning
    # -------------------------------------------------------------------------

    def _forward_obstacle(self, vehicle_depth: float, vehicle_world_x: float,
                          z_threshold: Optional[float] = None):
        """Return ``(peak_z, peak_world_x)`` of the shallowest manifold voxel
        in the centre-to-nose-plus-standoff window, or ``(None, nan)``.

        Scans manifold columns whose world-x lies in
        ``[vehicle_centre, nose + cliff_standoff]`` and whose depth
        ``z ≤ z_threshold`` (defaults to ``vehicle_z + safety_below_m``).
        The shallowest such voxel is the obstacle the avoidance latch must
        rise above and clear.

        No plateau or cliff-top guards — the latch's ratchet semantics
        (target_z ratchets shallower, target_x ratchets forward, release
        only when the vehicle has cleared past the tracked obstacle x)
        produce the right staircase behavior for continuous slopes and
        the right ascend-and-clear behavior for discrete cliffs.
        """
        c = self.cfg
        nose_x = vehicle_world_x + c.vehicle_length / 2.0
        x_min = vehicle_world_x           # vehicle centre
        x_max = nose_x + c.cliff_standoff  # nose + 2m
        z_max = (z_threshold if z_threshold is not None
                 else vehicle_depth + c.safety_below_m)

        ix_lo = max(self.cx,
                    int(np.floor((x_min - self.grid_origin_x) / c.dx)))
        ix_hi = min(self.nx,
                    int(np.ceil((x_max - self.grid_origin_x) / c.dx)) + 1)

        peak_z = np.inf
        peak_ix = -1
        for ix in range(ix_lo, ix_hi):
            if not self.manifold_observed[ix]:
                continue
            col_x = self.grid_to_world_x(ix)
            if col_x < x_min or col_x > x_max:
                continue
            z = self.manifold_z[ix]
            if z > z_max:
                continue
            if z < peak_z:
                peak_z = z
                peak_ix = ix
        if peak_ix < 0:
            return None, np.nan
        return peak_z, self.grid_to_world_x(peak_ix)

    def _safety_tail_blocked(self, vehicle_depth: float) -> bool:
        """Tail clearance check: centre to 2m behind the tail, 1m below.

        Fires when any observed manifold column has its world-x in
        ``[tail_x - safety_standoff_m, v_x]`` AND its manifold z in
        ``[vehicle_depth, vehicle_depth + safety_below_m)``.

        The window spans from the vehicle centre back to 2m behind the tail
        (tail = centre - vehicle_length/2).  When it fires, descent-in-place
        is unsafe — the vehicle would drive into terrain below or behind it.
        """
        c = self.cfg
        v_x = self.grid_to_world_x(self.cx)
        tail_x = v_x - c.vehicle_length / 2.0
        x_min = tail_x - c.safety_standoff_m  # 2m behind tail
        x_max = v_x                            # vehicle centre
        z_lo = vehicle_depth
        z_hi = vehicle_depth + c.safety_below_m

        ix_lo = max(0, int(np.floor((x_min - self.grid_origin_x) / c.dx)))
        ix_hi = min(self.nx, int(np.ceil((x_max - self.grid_origin_x) / c.dx)) + 1)

        for ix in range(ix_lo, ix_hi):
            if not self.manifold_observed[ix]:
                continue
            z = self.manifold_z[ix]
            if not (z_lo <= z < z_hi):
                continue
            x = self.grid_to_world_x(ix)
            if x_min <= x <= x_max:
                return True
        return False

    def build_commanded_depth(self, vehicle_depth: float):
        """
        Generate a commanded depth profile over the occupancy grid.

        Pipeline:

          Step 1 — Altitude-following baseline + DVL override at vehicle column.
            ``cmd_depth[ix] = manifold[ix] - imaging_altitude``.  At the
            vehicle column ``cx``, override with the min-across-beams DVL
            altitude reading so the altimeter + forward Janus give earliest
            terrain response.

          Step 2 — Cliff-top latch.
            Approach: climb_target caps cmd_depth at ``peak_z - imaging``
            while the vehicle climbs to maintain ``imaging_altitude`` distance
            from upcoming terrain.
            Commit: when the vehicle has crossed over the highest detected
            voxel within ``cliff_standoff``, latch on.  Hold target depth
            (= peak_z - imaging) and fly forward
            ``cliff_standoff + vehicle_length`` m at constant depth so
            the tail clears the cliff edge.  During the forward-hold, if any
            DVL beam reads altitude < ``imaging_altitude``, ratchet target
            shallower to maintain that minimum (vehicle never descends in
            this phase).  Release after the forward-hold distance.

          Step 3 — Safety tail cap (always active).
            Caps cmd_depth at ``vehicle_depth`` when manifold is within
            ``safety_below_m`` below the vehicle AND within
            ``safety_standoff_m`` horizontal of the tail.

        Mode selection from final ``cmd_depth[cx] - vehicle_depth`` (when the
        cliff latch does not consume the cycle), using a Schmitt-like band via
        ``altitude_overshoot_hysteresis_m`` so small ``cmd_depth`` jitter does not
        flip OBSTACLE_CLEAR ↔ ALT_CORRECTION:
          OBSTACLE_CLEAR — vehicle below target (must ascend in place).
          ALT_CORRECTION — vehicle above target (must descend in place).
          ALT_FOLLOW    — within hysteresis-augmented band of target.
        Then, if ``_safety_tail_blocked`` and mode is ALT_FOLLOW or
        ALT_CORRECTION → TAIL_CLEAR (constant-depth forward flight).  Cliff
        latch paths never reach this hook — forward obstacle dominates.

        Args:
            vehicle_depth: Current vehicle depth (m).
        """
        c = self.cfg
        vehicle_world_x = (
            self.grid_origin_x + self.cx * c.dx + self._shift_accum
        )

        # ----- Step 1: altitude-following baseline + DVL min override -----
        for ix in range(self.nx):
            z = self.manifold_z[ix]
            if np.isnan(z):
                self.cmd_depth[ix] = np.nan
            else:
                self.cmd_depth[ix] = max(0.0, z - c.imaging_altitude)
        # Vehicle column uses min-across-beams DVL altitude — picks up the
        # shortest terrain clearance from any beam (altimeter + Janus
        # forward-look).  Janus forward hit on rising terrain becomes the
        # altitude signal, giving earliest descent response.
        if not np.isnan(self.dvl_altitude):
            dvl_cmd = max(0.0, vehicle_depth + self.dvl_altitude - c.imaging_altitude)
            self.cmd_depth[self.cx] = max(dvl_cmd, self.cmd_depth[self.cx])
        elif not self.manifold_observed[self.cx]:
            # No DVL lock and no real manifold observation at the vehicle column
            # (grid defaults to bottom depth).  Do not command a dive to the
            # depth-window floor — that forces ALT_CORRECTION with zero forward
            # motion.  Hold at current depth until DVL or occupancy sees seafloor.
            self.cmd_depth[self.cx] = max(0.0, vehicle_depth)

        # ----- Step 2a: release latch if past tracked obstacle + margin -----
        if (self._cliff_top_committed
                and vehicle_world_x >= self._cliff_top_release_x):
            self._cliff_top_committed = False
            self._cliff_top_target_z = np.nan
            self._cliff_top_target_x = np.nan
            self._cliff_top_release_x = np.nan
            self._cliff_top_commit_heading = np.nan

        # ----- Step 2b: forward-obstacle detection — engage/update latch -----
        # Any manifold voxel within cliff_standoff ahead of the NOSE with
        # depth ``z ≤ vehicle_z + safety_below_m`` is an obstacle.  The
        # latch tracks the shallowest such voxel ever seen (target_z
        # ratchets only shallower) and the farthest peak x (target_x
        # ratchets only forward).
        #
        # For a discrete cliff: detection fires while approaching, target
        # stays at the cliff peak.  Vehicle ascends to peak - imaging, then
        # forward at that depth until past peak + cliff_standoff +
        # vehicle_length, then release.
        #
        # For a continuous upslope: each cycle a new shallower far-edge
        # voxel enters the window, target_z ratchets shallower.  Vehicle
        # alternates between ascending (when vehicle_z > target_z) and
        # forward at constant depth (when vehicle_z ≤ target_z) —
        # the staircase pattern.  target_x ratchets forward each cycle so
        # release_x stays ahead of the vehicle as long as the slope keeps
        # rising.  When the slope plateaus, target stops updating, the
        # vehicle catches up to release_x, latch releases.
        peak_z, peak_x = self._forward_obstacle(vehicle_depth, vehicle_world_x)
        if peak_z is not None:
            # Don't clamp to 0 — if peak_z - imaging_altitude is negative
            # (obstacle top is above water), vehicle tries to ascend past
            # the surface but the simulator's z≥0 clamp holds it there.
            # This produces the correct stuck-at-surface behavior on a slope
            # too steep to traverse, instead of letting the vehicle slide
            # through terrain.
            new_target_z = peak_z - c.imaging_altitude
            if not self._cliff_top_committed:
                self._cliff_top_committed = True
                self._cliff_top_target_z = new_target_z
                self._cliff_top_target_x = peak_x
                self._cliff_top_commit_heading = self._last_vehicle_heading
            else:
                if new_target_z < self._cliff_top_target_z:
                    self._cliff_top_target_z = new_target_z
                if peak_x > self._cliff_top_target_x:
                    self._cliff_top_target_x = peak_x
                    # Re-anchor the commit heading: the latch now tracks a peak
                    # observed at the current heading, so the turn-away release
                    # must be measured from here, not from the original commit.
                    self._cliff_top_commit_heading = self._last_vehicle_heading
            self._cliff_top_release_x = (
                self._cliff_top_target_x
                + c.cliff_standoff + c.vehicle_length
            )
        elif self._cliff_top_committed:
            # Latch is committed but the narrow window (safety_below_m) found
            # nothing.  Run a wider scan using imaging_altitude as the depth
            # threshold to catch terrain between safety_below_m and
            # imaging_altitude below the vehicle.
            wide_z, wide_x = self._forward_obstacle(
                vehicle_depth, vehicle_world_x,
                z_threshold=vehicle_depth + c.imaging_altitude,
            )
            if wide_z is not None:
                new_target_z = wide_z - c.imaging_altitude
                if new_target_z < self._cliff_top_target_z:
                    self._cliff_top_target_z = new_target_z
                    if wide_x > self._cliff_top_target_x:
                        self._cliff_top_target_x = wide_x
                        self._cliff_top_commit_heading = (
                            self._last_vehicle_heading
                        )
                    self._cliff_top_release_x = (
                        self._cliff_top_target_x
                        + c.cliff_standoff + c.vehicle_length
                    )
                # else: terrain at or below imaging_altitude — keep latch so
                # the tail clears the highest obstacle voxel before the
                # altimeter takes over.
            else:
                # Both narrow and wide scans empty: no forward obstacle visible.
                # Only release if the vehicle has turned significantly from the
                # heading at latch commit (same threshold as clear_stale_voxels),
                # which is what causes the occupied voxels to disappear.  Without
                # this guard the cliff peak passing behind vehicle_center during
                # normal OBSTACLE_HOLD forward flight would trigger a premature
                # release before the tail has cleared.
                h_cur = self._last_vehicle_heading
                h_cmt = self._cliff_top_commit_heading
                if not (np.isnan(h_cur) or np.isnan(h_cmt)):
                    # Same wrap handling as clear_stale_voxels: the modulo
                    # first keeps this correct for unwrapped headings, where
                    # a bare (2*pi - diff) would go negative.
                    diff = abs(h_cur - h_cmt) % (2.0 * np.pi)
                    diff = min(diff, 2.0 * np.pi - diff)
                    if diff > np.radians(c.stale_heading_threshold_deg):
                        self._cliff_top_committed = False
                        self._cliff_top_target_z = np.nan
                        self._cliff_top_target_x = np.nan
                        self._cliff_top_release_x = np.nan
                        self._cliff_top_commit_heading = np.nan

        # ----- Step 2c: apply latch (effective cmd_depth) -----
        # Two release paths: Step 2a (x-based: tail has cleared peak + standoff)
        # and Step 2b early-release (both scans empty: vehicle turned away).
        # A narrow-scan miss alone does NOT release — the obstacle may have just
        # exited the narrow window while the vehicle is still approaching the peak.
        # Always use the committed target depth exactly so OBSTACLE_HOLD holds
        # constant depth while the vehicle flies forward over the obstacle.
        if self._cliff_top_committed:
            effective = self._cliff_top_target_z
            for ix in range(self.cx, self.nx):
                self.cmd_depth[ix] = effective
            self._set_mode_from_cmd_depth(vehicle_depth)
            # If vehicle is already at/above target, fly forward at target depth.
            if self.control_mode != 'OBSTACLE_CLEAR':
                self.control_mode = 'OBSTACLE_HOLD'
            return

        # ----- Step 3: safety tail cap (always active) -----
        if self._safety_tail_blocked(vehicle_depth):
            for ix in range(self.cx, self.nx):
                if self.cmd_depth[ix] > vehicle_depth:
                    self.cmd_depth[ix] = vehicle_depth

        # ----- Step 4: propagate vehicle-column depth to unobserved ahead columns -----
        # Unobserved columns default to grid-bottom manifold depth, which drives
        # cmd_depth deep.  The motion step interpolates one dt ahead to get its
        # target_z; without this propagation that look-ahead pulls the vehicle
        # down even when dvl_altitude correctly commands a climb.  Hold the
        # vehicle column's cmd_depth where the manifold is not yet observed.
        for ix in range(self.cx + 1, self.nx):
            if not self.manifold_observed[ix]:
                self.cmd_depth[ix] = self.cmd_depth[self.cx]

        # ----- Mode selection -----
        self._set_mode_from_cmd_depth(vehicle_depth)
        # Tail clearance overrides altitude follow / correction only; forward
        # obstacle latch (early return above) and OBSTACLE_CLEAR are unchanged.
        if (self._safety_tail_blocked(vehicle_depth)
                and self.control_mode in ('ALT_FOLLOW', 'ALT_CORRECTION')):
            self.control_mode = 'TAIL_CLEAR'

    def _set_mode_from_cmd_depth(self, vehicle_depth: float):
        """Pick mode from ``cmd_depth[cx] - vehicle_depth`` using hysteresis.

        ``dz = cmd_depth[cx] - vehicle_depth``: negative means vehicle is deeper
        than commanded (needs OBSTACLE_CLEAR climb); positive means shallower
        than commanded (needs ALT_CORRECTION dive).

        Entries from ALT_FOLLOW use ``altitude_overshoot_threshold_m``; exits
        from OBSTACLE_CLEAR / ALT_CORRECTION use the tighter interior band
        ``altitude_overshoot_threshold_m - altitude_overshoot_hysteresis_m``.
        """
        c = self.cfg
        target_z = self.cmd_depth[self.cx]
        if np.isnan(target_z):
            self.control_mode = "ALT_FOLLOW"
            return

        dz = float(target_z) - float(vehicle_depth)
        T = float(c.altitude_overshoot_threshold_m)
        h_raw = float(c.altitude_overshoot_hysteresis_m)
        h = np.clip(h_raw, 0.0, max(0.0, T - 1e-6))
        inside = max(0.0, T - h)
        prev = self.control_mode

        obstacle_clear = (
            dz <= -T
            or (prev == "OBSTACLE_CLEAR" and dz < -inside)
        )
        altitude_correction = (
            dz >= T
            or (prev == "ALT_CORRECTION" and dz > inside)
        )

        # Forward obstacle dominates when both widen — pick the stronger ascent need.
        if obstacle_clear and altitude_correction:
            if dz <= -inside:
                self.control_mode = "OBSTACLE_CLEAR"
            elif dz >= inside:
                self.control_mode = "ALT_CORRECTION"
            else:
                self.control_mode = "ALT_FOLLOW"
            return

        if obstacle_clear:
            self.control_mode = "OBSTACLE_CLEAR"
            return

        if altitude_correction:
            self.control_mode = "ALT_CORRECTION"
            return

        self.control_mode = "ALT_FOLLOW"

    def build_path_waypoints(self) -> list:
        """
        Convert the commanded depth profile into a waypoint polyline with
        explicit vertical segments at cliff transitions.

        Returns:
            List of (world_x, depth_z) tuples defining the path polyline.
            Vertical segments appear as two waypoints at the same X with
            different Z values.
        """
        c = self.cfg
        step_thresh = 2 * c.dz
        waypoints = []

        for ix in range(self.cx, self.nx):
            world_x = self.grid_to_world_x(ix)
            depth = self.cmd_depth[ix]

            if np.isnan(depth):
                continue

            if len(waypoints) == 0:
                waypoints.append((world_x, depth))
                continue

            prev_x, prev_z = waypoints[-1]
            dz = abs(depth - prev_z)

            if dz > step_thresh:
                if abs(world_x - prev_x) > 0.01:
                    waypoints.append((world_x, prev_z))
                waypoints.append((world_x, depth))
            else:
                waypoints.append((world_x, depth))

        self.path_waypoints = waypoints
        return waypoints

    def get_commanded_depth_at_vehicle(self) -> float:
        """
        Returns the commanded depth at the vehicle's current grid position.
        """
        if 0 <= self.cx < self.nx:
            return self.cmd_depth[self.cx]
        return np.nan

    # -------------------------------------------------------------------------
    # Full update cycle
    # -------------------------------------------------------------------------

    def clear_stale_voxels(self, vehicle_heading: float) -> None:
        """Reset any observed voxel whose observation heading is too far from current heading.

        ``voxel_heading`` records the vehicle heading at the time of the most
        recent observation of each voxel, whether that observation was a hit
        (occupied) or a miss (free).  Any voxel whose stored heading differs
        from *vehicle_heading* by more than ``stale_heading_threshold_deg`` is
        reset to prior (unobserved) probability and its heading cleared.

        Exception: occupied voxels in the columns directly under the vehicle
        footprint have their stored heading refreshed to the current heading
        instead of being cleared.  These are real terrain observations the
        vehicle is flying over — discarding them on a heading change would
        cause the planner to lose the seafloor directly below.  Free voxels
        under the footprint are still cleared when stale, since a free-space
        observation from one heading may not hold from another.

        Args:
            vehicle_heading: Current vehicle heading (radians, same convention
                             as headings stored by the sensor update methods).
        """
        if np.isnan(vehicle_heading):
            return
        c = self.cfg

        # Refresh heading for occupied voxels under the vehicle body so they
        # are never cleared by the stale gate.
        half_bins = int(np.ceil(c.vehicle_length / (2.0 * c.dx)))
        ix_lo = max(0, self.cx - half_bins)
        ix_hi = min(self.nx, self.cx + half_bins + 1)
        footprint_occupied = self.grid[:, ix_lo:ix_hi] >= c.occ_thresh
        self.voxel_heading[:, ix_lo:ix_hi][footprint_occupied] = vehicle_heading

        threshold = np.radians(c.stale_heading_threshold_deg)
        diff = np.abs(self.voxel_heading - vehicle_heading) % (2.0 * np.pi)
        diff = np.minimum(diff, 2.0 * np.pi - diff)
        stale = (~np.isnan(self.voxel_heading)) & (diff > threshold)
        self.grid[stale] = c.prior
        self.voxel_heading[stale] = np.nan

    def update(self, vehicle_depth: float, vehicle_heading: float = np.nan):
        """
        Run the full processing pipeline: stale-voxel clearing, manifold
        extraction, path planning, and waypoint generation. Call this after
        sensor updates.

        Args:
            vehicle_depth:   Current vehicle depth (m).
            vehicle_heading: Current vehicle heading (radians).  When provided,
                             occupied voxels whose observation heading differs by
                             more than ``stale_heading_threshold_deg`` are reset
                             to prior before planning.

        Returns:
            Commanded depth at the vehicle position.
        """
        self._last_vehicle_heading = vehicle_heading
        self.clear_stale_voxels(vehicle_heading)
        self.build_cliff_manifold()
        self.build_commanded_depth(vehicle_depth)
        self.build_path_waypoints()
        return self.get_commanded_depth_at_vehicle()

    # -------------------------------------------------------------------------
    # Inspection / debug helpers
    # -------------------------------------------------------------------------

    def get_debug_summary(self, vehicle_depth: float) -> str:
        """
        Return a human-readable multi-line diagnostic string for the current
        planner state.  Useful for diagnosing stuck / oscillation conditions.

        Args:
            vehicle_depth: Current vehicle depth (m).

        Returns:
            Formatted string ready to print or log.
        """
        c = self.cfg
        cx = self.cx
        climb_bins = int(np.ceil(c.cliff_standoff / c.dx))
        safety_bins = int(np.ceil(
            (c.safety_standoff_m + c.vehicle_length / 2.0) / c.dx))
        raw = self.manifold_z
        cmd = self.cmd_depth
        win = 6

        def fmt(arr, col):
            if 0 <= col < self.nx:
                v = arr[col]
                if np.isnan(v):
                    return '   nan'
                return f'{v:+6.2f}'
            return '  ---'

        idxs = range(max(0, cx - win), min(self.nx, cx + win + 1))
        hdr  = '  '.join(f'{"cx"+("" if i==cx else str(i-cx)):>6}' for i in idxs)
        mrow = '  '.join(fmt(raw, i) for i in idxs)
        crow = '  '.join(fmt(cmd, i) for i in idxs)
        xrow = '  '.join(f'{self.grid_to_world_x(i):+6.1f}' for i in idxs)

        # Forward-obstacle detection within cliff_standoff ahead of nose.
        vehicle_world_x = self.grid_origin_x + self.cx * c.dx + self._shift_accum
        peak_z, _ = self._forward_obstacle(vehicle_depth, vehicle_world_x)
        climb_target = (max(0.0, peak_z - c.imaging_altitude)
                        if peak_z is not None else None)

        # Safety tail-check: first behind column whose manifold segment crosses
        # the depth band [v_z, v_z + safety_below_m) within
        # (safety_standoff_m + vehicle_length/2) behind the center.
        z_lo = vehicle_depth
        z_hi = vehicle_depth + c.safety_below_m
        safety_ix = None
        for ix in range(max(0, cx - safety_bins), cx):
            if ix + 1 >= self.nx:
                continue
            if not (self.manifold_observed[ix] and self.manifold_observed[ix + 1]):
                continue
            s_lo = min(raw[ix], raw[ix + 1])
            s_hi = max(raw[ix], raw[ix + 1])
            if s_hi >= z_lo and s_lo < z_hi:
                safety_ix = ix
                break

        wps = self.path_waypoints[:6]
        wp_str = '  '.join(f'({w[0]:+.1f},{w[1]:.1f})' for w in wps)

        dvl_str = (
            f"{self.dvl_altitude:.3f}m"
            if not np.isnan(self.dvl_altitude)
            else "nan"
        )
        lines = [
            f"  mode={self.control_mode:<15}  "
            f"dvl_alt={dvl_str}  target={c.imaging_altitude:.2f}m  "
            f"veh_z={vehicle_depth:.3f}m  cmd@cx={cmd[cx]:.3f}m",
            f"  col offset: {hdr}",
            f"  world_x:    {xrow}",
            f"  manifold_z: {mrow}",
            f"  cmd_depth:  {crow}",
            f"  climb:  " + (f"target={climb_target:.2f}m" if climb_target is not None
                              else "no rise in window"),
            f"  safety: "
            + (f"manifold in band at col cx{safety_ix-cx} "
               f"(x={self.grid_to_world_x(safety_ix):+.2f}m)"
               if safety_ix is not None else "clear"),
            f"  waypoints[0:6]: {wp_str if wp_str else '(none)'}",
        ]
        return '\n'.join(lines)

    def get_grid_snapshot(self) -> dict:
        """
        Return a snapshot of the current state for visualization/debugging.
        """
        return {
            'grid': self.grid.copy(),
            'nx': self.nx,
            'nz': self.nz,
            'cx': self.cx,
            'dx': self.cfg.dx,
            'dz': self.cfg.dz,
            'grid_origin_x': self.grid_origin_x,
            'manifold_grid_origin_x': self.manifold_grid_origin_x,
            'z_min': self.grid_origin_z,
            'z_max': self.grid_origin_z + self.nz * self.cfg.dz,
            'manifold_iz': self.manifold_iz.copy(),
            'manifold_z': self.manifold_z.copy(),
            'cmd_depth': self.cmd_depth.copy(),
            'path_waypoints': list(self.path_waypoints),
            'dvl_altitude': float(self.dvl_altitude) if not np.isnan(self.dvl_altitude) else None,
            'control_mode': self.control_mode,
        }


# ===========================================================================
# High-level sensor interface
# ===========================================================================

class ObstacleMapper:
    """Thread-safe AUV obstacle avoidance interface.

    Accepts asynchronous sensor measurements from DVL, altimeter, and forward
    sonar via update_sensor(). Advances the occupancy grid based on the forward
    displacement between consecutive poses, accounting for vehicle heading.

    Velocity commands suitable for a real AUV controller are returned by
    get_control(). Vehicle altitude (minimum of last DVL and altimeter vertical
    readings) is returned by get_altitude().

    The forward sonar origin is at the vehicle nose (vehicle_center +
    vehicle_length/2). Sonar ranges are measured from that point.

    Args:
        config:            Occupancy map and path-planning parameters.
        dvl_config:        DVL beam geometry and max range.
        sonar_config:      Forward sonar beam width and max range.
        altimeter_config:  Downward altimeter max range (optional).
    """

    def __init__(
        self,
        config: OccupancyMapConfig,
        dvl_config: DVLConfig,
        sonar_config: SonarConfig,
        altimeter_config: Optional[AltimeterConfig] = None,
    ):
        self._omap = OccupancyMap(config)
        self._dvl_config = dvl_config
        self._sonar_config = sonar_config
        self._altimeter_config = altimeter_config or AltimeterConfig()
        self._lock = threading.Lock()
        self._last_pose: Optional[Pose] = None
        self._altimeter_altitude: float = np.nan

    @property
    def omap(self) -> OccupancyMap:
        """Underlying OccupancyMap for visualization and debug access."""
        return self._omap

    def reset(self, pose: Pose) -> None:
        """Reset map state, re-centering on the given pose."""
        with self._lock:
            self._omap.reset(0.0, pose.depth)
            self._last_pose = pose
            self._altimeter_altitude = np.nan

    def update_sensor(
        self,
        sensor_type: SensorType,
        measurement: Union[DVLMeasurement, AltimeterMeasurement, SonarMeasurement],
        pose: Pose,
    ) -> None:
        """Process a sensor measurement at the given vehicle pose.

        Advances the occupancy grid by the forward component of displacement
        from the previous pose (using the previous pose's heading for
        projection), applies the sensor observation, and updates the avoidance
        plan. Thread-safe; safe to call from concurrent sensor callbacks.

        Args:
            sensor_type:  Which sensor produced the measurement.
            measurement:  DVLMeasurement, AltimeterMeasurement, or SonarMeasurement.
            pose:         Current vehicle pose in NED frame.
        """
        with self._lock:
            self._advance_to_pose(pose)
            fwd_x = self._vehicle_forward_x()

            if sensor_type == SensorType.DVL:
                self._omap.update_dvl_ray(
                    measurement.ranges,
                    self._dvl_config.beam_angles_rad,
                    pose.depth,
                    fwd_x,
                    hit_surface=measurement.hit_surface,
                    vehicle_heading=pose.heading,
                    can_clear=self._dvl_config.beam_can_clear,
                )

            elif sensor_type == SensorType.ALTIMETER:
                valid = (
                    measurement.hit
                    and measurement.range_m < self._altimeter_config.max_range - 0.05
                )
                self._altimeter_altitude = measurement.range_m if valid else np.nan
                self._omap.update_altimeter_ray(
                    measurement.range_m,
                    pose.depth,
                    fwd_x,
                    measurement.hit,
                    vehicle_heading=pose.heading,
                )

            elif sensor_type == SensorType.SONAR:
                nose_x = fwd_x + self._omap.cfg.vehicle_length / 2.0
                self._omap.update_sonar(
                    measurement.range_m,
                    self._sonar_config.half_angle_rad,
                    pose.depth,
                    nose_x,
                    measurement.hit,
                    vehicle_heading=pose.heading,
                )

            self._omap.update(pose.depth, pose.heading)

    def update_pose(self, pose: Pose) -> None:
        """Advance the map to the current vehicle pose and re-run planning.

        Use this at the control rate to keep the map and plan current between
        sensor firings.  No occupancy data is written — the grid origin shifts
        via advance() / shift_depth() and the full planning pipeline re-runs
        (stale clearing, manifold, depth profile, waypoints).

        Safe to call when no sensor has fired yet, or when called in the same
        step as a sensor update (the zero-displacement advance is a no-op).

        Thread-safe.
        """
        with self._lock:
            self._advance_to_pose(pose)
            self._omap.update(pose.depth, pose.heading)

    def get_control(self) -> ControlCommand:
        """Return the current obstacle avoidance command.

        Priority: forward obstacle (OBSTACLE_CLEAR / OBSTACLE_HOLD from latch or
        below-target ascent) unchanged; tail blocked with ALT_FOLLOW or
        ALT_CORRECTION → TAIL_CLEAR (survey_speed, DEPTH_HOLD at present depth).

        ALT_FOLLOW / ALT_CORRECTION otherwise:
            vertical_mode  = 'ALT_FOLLOW'
            vertical_target = imaging_altitude (m above seafloor)
            vx             = survey_speed or 0 (ALT_CORRECTION)

        OBSTACLE_CLEAR: DEPTH_HOLD at cmd depth, vx=0 until within band.

        OBSTACLE_HOLD / TAIL_CLEAR: DEPTH_HOLD, vx=survey_speed.

        Thread-safe.
        """
        with self._lock:
            if self._last_pose is None:
                return ControlCommand(
                    vx=self._omap.cfg.survey_speed,
                    vertical_mode='ALT_FOLLOW',
                    vertical_target=self._omap.cfg.imaging_altitude,
                )
            mode = self._omap.control_mode
            c = self._omap.cfg
            if mode == 'OBSTACLE_CLEAR':
                cmd_depth = self._omap.get_commanded_depth_at_vehicle()
                target_depth = (cmd_depth
                                if not np.isnan(cmd_depth)
                                else self._last_pose.depth)
                dz = target_depth - self._last_pose.depth
                vx = 0.0 if abs(dz) > 0.1 else c.survey_speed
                return ControlCommand(
                    vx=vx,
                    vertical_mode='DEPTH_HOLD',
                    vertical_target=target_depth,
                )
            if mode == 'OBSTACLE_HOLD':
                cmd_depth = self._omap.get_commanded_depth_at_vehicle()
                target_depth = (cmd_depth
                                if not np.isnan(cmd_depth)
                                else self._last_pose.depth)
                return ControlCommand(
                    vx=c.survey_speed,
                    vertical_mode='DEPTH_HOLD',
                    vertical_target=target_depth,
                )
            if mode == 'ALT_CORRECTION':
                cmd_depth = self._omap.get_commanded_depth_at_vehicle()
                target = (cmd_depth
                          if not np.isnan(cmd_depth)
                          else self._last_pose.depth)
                return ControlCommand(
                    vx=0.0,
                    vertical_mode='DEPTH_HOLD',
                    vertical_target=target,
                )
            if mode == 'TAIL_CLEAR':
                # Constant-depth forward until tail safety band clears;
                # overrides ALT_FOLLOW / ALT_CORRECTION when not in forward
                # obstacle ascent (OBSTACLE_CLEAR) or latch hold.
                return ControlCommand(
                    vx=c.survey_speed,
                    vertical_mode='DEPTH_HOLD',
                    vertical_target=self._last_pose.depth,
                )
            # ALT_FOLLOW — altitude controller handles heave
            return ControlCommand(
                vx=c.survey_speed,
                vertical_mode='ALT_FOLLOW',
                vertical_target=c.imaging_altitude,
            )

    def get_altitude(self) -> float:
        """Vehicle altitude above seafloor (m) from the most recent sensor readings.

        Returns the minimum of the last DVL vertical range and the last
        altimeter range.  Both values are derived directly from sensor
        measurements and are independent of the voxel map, so this reading
        remains valid even when the occupancy grid has been cleared by the
        stale-heading gate.

        Returns NaN when neither sensor has produced a valid return yet.
        Thread-safe.
        """
        with self._lock:
            candidates = [
                v for v in (self._omap.dvl_altitude, self._altimeter_altitude)
                if not np.isnan(v)
            ]
            return min(candidates) if candidates else np.nan

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_to_pose(self, pose: Pose) -> None:
        """Advance the grid to match a new pose. Must be called with lock held."""
        if self._last_pose is None:
            self._omap.reset(0.0, pose.depth)
            self._last_pose = pose
            return
        dn = pose.north - self._last_pose.north
        de = pose.east - self._last_pose.east

        # Detect mission restart / teleport: if the straight-line jump exceeds
        # the grid length the vehicle is outside the mapped area.  Reset rather
        # than projecting a huge (and mis-signed) ds onto the arc-local axis.
        grid_len = (self._omap.nx - 1) * self._omap.cfg.dx
        if dn * dn + de * de > grid_len * grid_len:
            self._omap.reset(0.0, pose.depth)
            self._last_pose = pose
            return

        h = self._last_pose.heading
        ds = dn * np.cos(h) + de * np.sin(h)
        if ds > 0.0:
            self._omap.advance(ds)
        self._omap.shift_depth(pose.depth)
        self._last_pose = pose

    def _vehicle_forward_x(self) -> float:
        """Along-track position of the vehicle in the occupancy map frame."""
        omap = self._omap
        return omap.grid_origin_x + omap.cx * omap.cfg.dx + omap._shift_accum

