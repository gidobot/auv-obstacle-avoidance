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
this manifold using a three-mode strategy:

  Mode: ALT_FOLLOW  (default)
    Track imaging_altitude above the raw manifold (terrain-following).

  Mode: OBSTACLE_CLEAR  (forward obstacle or cliff detected)
    Begin climbing cliff_standoff metres before the obstacle face.
    Rise to imaging_altitude above the shallowest observed obstacle voxel.
    Hold that depth (depth-control) for cliff_standoff metres past the
    last elevated obstacle column so the altimeter can lock on the new
    surface, then return to ALT_FOLLOW on top of the obstacle/cliff.

  Mode: ALT_CORRECTION  (altitude diverges above imaging altitude target)
    Triggered when vehicle's altitude exceeds imaging altitude by more
    than ``altitude_overshoot_threshold_m``.  Vehicle stops forward motion
    and descends in place toward imaging altitude.  Safety check still
    runs and forces horizontal motion when the tail is too close to the
    terrain behind.  Naturally handles cliff descents (altitude shoots up
    on cliff edge) and steep slopes (altitude drifts up when the slope
    descends faster than vertical_speed × dt per forward step).

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

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


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

    # Safety tail check (always active).  Caps cmd_depth at vehicle_depth
    # whenever manifold is found within ``safety_below_m`` below the vehicle
    # and within ``safety_standoff_m`` horizontal of the tail.  Prevents the
    # tail from skimming terrain in any mode.
    safety_standoff_m: float = 1.0    # Horizontal clearance behind tail (m)
    safety_below_m: float = 1.0       # Depth band below vehicle (m)

    # Occupancy probability parameters
    prior: float = 0.5            # Prior occupancy probability
    occ_thresh: float = 0.62      # Threshold to consider a voxel occupied

    # DVL observation model
    dvl_hit_prob: float = 0.14    # P(occupied | hit) increment
    dvl_miss_prob: float = 0.07   # P(free | miss) decrement
    dvl_max_occ: float = 0.98     # Max occupancy from DVL
    dvl_min_occ: float = 0.02     # Min occupancy from DVL

    # Forward sonar observation model (noisier, lower confidence)
    sonar_hit_prob: float = 0.04  # P(occupied | hit) increment
    sonar_miss_prob: float = 0.02 # P(free | miss) decrement
    sonar_max_occ: float = 0.92   # Max occupancy from sonar
    sonar_min_occ: float = 0.05   # Min occupancy from sonar


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

        # Spatially-binned heading-change accumulator (1-D, same nx as grid).
        # Each column accumulates the total heading change (radians) that
        # occurred while the vehicle occupied that spatial bin.  Slides left
        # with the grid via advance() and is reset to zero in reset().
        # Used by Simulator3D for the turn-mirror sliding-window check.
        self.turn_dh_bins: np.ndarray = np.zeros(self.nx, dtype=np.float64)

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
        # hitting beams, updated each cycle by update_dvl_ray.  Used as the
        # altitude-following reference at the vehicle's current column instead
        # of the occupancy-filtered manifold, giving faster terrain response.
        self.dvl_altitude: float = np.nan

        # Current control mode at vehicle position (for display / logging).
        self.control_mode: str = "ALT_FOLLOW"

        # Cliff-top crossing latch.  Set when the vehicle crosses over the
        # highest detected voxel within cliff_standoff ahead during a climb,
        # cleared after the vehicle has flown
        # (cliff_standoff + vehicle_length + 1 m) past the peak at constant
        # depth (tail-clearance distance).
        self._cliff_top_committed: bool = False
        self._cliff_top_target_z: float = np.nan  # ratchets shallower only
        self._cliff_top_target_x: float = np.nan  # ratchets forward only
        self._cliff_top_release_x: float = np.nan

    def reset(self, vehicle_world_x: float = 0.0, vehicle_depth: float = 0.0):
        """Reset the grid to prior probability and re-center on vehicle."""
        c = self.cfg
        self.grid[:, :] = c.prior
        self.turn_dh_bins[:] = 0.0
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
            self.turn_dh_bins[:] = 0.0
        else:
            self.grid[:, :-cols] = self.grid[:, cols:]
            self.grid[:, -cols:] = self.cfg.prior
            self.turn_dh_bins[:-cols] = self.turn_dh_bins[cols:]
            self.turn_dh_bins[-cols:] = 0.0

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
            else:
                self.grid[:-rows, :] = self.grid[rows:, :]
                self.grid[-rows:, :] = self.cfg.prior
        else:
            # Vehicle moved shallower — drop deep rows, expose new shallow rows
            r = -rows
            if r >= self.nz:
                self.grid[:, :] = self.cfg.prior
            else:
                self.grid[r:, :] = self.grid[:-r, :]
                self.grid[:r, :] = self.cfg.prior

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
    ):
        """
        Update occupancy by ray-marching each DVL beam, and track direct altitude.

        Marks cells along the beam as free, and the endpoint cell as occupied.
        This provides more information per observation than endpoint-only updates.

        Also computes self.dvl_altitude — the minimum vertical-axis range
        component across all beams that actually hit the seafloor:

            dvl_altitude = min( range_i * cos(angle_i) )   for hit beams

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
            ang = beam_angles[i]

            # A beam that did not hit real terrain is a max-range return.
            # Treat the entire beam path (including the endpoint) as free space.
            is_hit = (hit_surface is None) or bool(hit_surface[i])

            r = range_step
            while r < r_max - range_step:
                dx = np.sin(ang) * r
                dz = np.cos(ang) * r
                ix, iz = self.world_to_grid(vehicle_world_x + dx,
                                            vehicle_depth + dz)
                if self._in_bounds(ix, iz):
                    self.grid[iz, ix] = max(c.dvl_min_occ,
                                            self.grid[iz, ix] - c.dvl_miss_prob)
                r += range_step

            dx = np.sin(ang) * r_max
            dz = np.cos(ang) * r_max
            ix, iz = self.world_to_grid(vehicle_world_x + dx,
                                        vehicle_depth + dz)
            if self._in_bounds(ix, iz):
                if is_hit:
                    self.grid[iz, ix] = min(c.dvl_max_occ,
                                            self.grid[iz, ix] + c.dvl_hit_prob)
                else:
                    # No real return — end of beam is also free.
                    self.grid[iz, ix] = max(c.dvl_min_occ,
                                            self.grid[iz, ix] - c.dvl_miss_prob)

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

    def update_sonar(
        self,
        sonar_range: float,
        sonar_half_angle: float,
        vehicle_depth: float,
        vehicle_world_x: float,
        hit_obstacle: bool,
        range_step: float = 0.2,
        angle_steps: int = 7,
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
        c = self.cfg
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
                r += range_step

            if hit_obstacle:
                dx = sonar_range * np.cos(ang)
                dz = sonar_range * np.sin(ang)
                ix, iz = self.world_to_grid(vehicle_world_x + dx,
                                            vehicle_depth + dz)
                if self._in_bounds(ix, iz):
                    self.grid[iz, ix] = min(c.sonar_max_occ,
                                            self.grid[iz, ix] + c.sonar_hit_prob)

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

    def _forward_obstacle(self, vehicle_depth: float, vehicle_world_x: float):
        """Return ``(peak_z, peak_world_x)`` of the shallowest manifold voxel
        in the nose-forward standoff window, or ``(None, nan)``.

        Scans manifold columns whose world-x lies in ``[nose, nose +
        cliff_standoff]`` and whose depth ``z ≤ vehicle_z + safety_below_m``
        (within ``safety_below_m`` below the vehicle, or shallower).  The
        shallowest such voxel is "the obstacle" the avoidance latch must
        rise above and clear.

        No plateau or cliff-top guards — the latch's ratchet semantics
        (target_z ratchets shallower, target_x ratchets forward, release
        only when the vehicle has cleared past the tracked obstacle x)
        produce the right staircase behavior for continuous slopes and
        the right ascend-and-clear behavior for discrete cliffs.
        """
        c = self.cfg
        nose_x = vehicle_world_x + c.vehicle_length / 2.0
        x_min = nose_x
        x_max = nose_x + c.cliff_standoff
        z_max = vehicle_depth + c.safety_below_m

        ix_lo = max(self.cx + 1,
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
        """Always-on tail clearance check.

        Fires when any observed manifold COLUMN behind the vehicle has its
        z value in the band ``[vehicle_depth, vehicle_depth + safety_below_m)``
        AND its world-x in the standoff zone
        ``[v_x - (safety_standoff_m + vehicle_length/2), v_x)``.

        Used as a depth cap: if it fires, ``cmd_depth`` from ``cx`` onward is
        capped at ``vehicle_depth``, preventing the vehicle from diving
        deeper while still allowing alt-follow's upward commands (shallower
        cmd_depth) to pass through.

        Column-based (not segment-interpolated) on purpose: a cliff face
        segment's z range linearly spans [cliff_top, cliff_base] and a
        segment-based check would interpret this as "manifold close to tail
        throughout the descent."  In reality the cliff face is at one
        column (the edge), the tail's x has already passed it, and the
        descent should be allowed.  Column-based avoids that false positive.
        Catches slopes whose per-bin drop is < safety_below_m (i.e., slope
        angle < ~atan(safety_below_m / dx) ≈ 63° at defaults).  Steeper
        terrain falls under ALT_CORRECTION via altitude divergence.
        """
        c = self.cfg
        standoff = c.safety_standoff_m + c.vehicle_length / 2.0
        look_behind_bins = int(np.ceil(standoff / c.dx))
        z_lo = vehicle_depth
        z_hi = vehicle_depth + c.safety_below_m
        v_x = self.grid_to_world_x(self.cx)
        standoff_x_min = v_x - standoff
        start = max(0, self.cx - look_behind_bins)
        for ix in range(start, self.cx):
            if not self.manifold_observed[ix]:
                continue
            z = self.manifold_z[ix]
            if not (z_lo <= z < z_hi):
                continue
            x = self.grid_to_world_x(ix)
            if standoff_x_min <= x < v_x:
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
            ``cliff_standoff + vehicle_length + 1`` m at constant depth so
            the tail clears the cliff edge.  During the forward-hold, if any
            DVL beam reads altitude < ``imaging_altitude``, ratchet target
            shallower to maintain that minimum (vehicle never descends in
            this phase).  Release after the forward-hold distance.

          Step 3 — Safety tail cap (always active).
            Caps cmd_depth at ``vehicle_depth`` when manifold is within
            ``safety_below_m`` below the vehicle AND within
            ``safety_standoff_m`` horizontal of the tail.

        Mode selection from final ``cmd_depth[cx] - vehicle_depth``:
          OBSTACLE_CLEAR — vehicle below target (must ascend in place).
          ALT_CORRECTION — vehicle above target (must descend in place).
          ALT_FOLLOW    — within ``altitude_overshoot_threshold_m`` of target.

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
            self.cmd_depth[self.cx] = max(
                0.0, vehicle_depth + self.dvl_altitude - c.imaging_altitude
            )

        # ----- Step 2a: release latch if past tracked obstacle + margin -----
        if (self._cliff_top_committed
                and vehicle_world_x >= self._cliff_top_release_x):
            self._cliff_top_committed = False
            self._cliff_top_target_z = np.nan
            self._cliff_top_target_x = np.nan
            self._cliff_top_release_x = np.nan

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
        # vehicle_length + 1, then release.
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
            else:
                if new_target_z < self._cliff_top_target_z:
                    self._cliff_top_target_z = new_target_z
                if peak_x > self._cliff_top_target_x:
                    self._cliff_top_target_x = peak_x
            self._cliff_top_release_x = (
                self._cliff_top_target_x
                + c.cliff_standoff + c.vehicle_length + 1.0
            )

        # ----- Step 2c: apply latch (effective cmd_depth) -----
        if self._cliff_top_committed:
            # Vehicle should never descend during avoidance.  Hold at
            # min(target_z, vehicle_z): if vehicle is deeper than target,
            # cmd = target (forces ascent via OBSTACLE_CLEAR).  If vehicle
            # has already overshot shallower than target, cmd = vehicle_z
            # so dz=0 and vehicle flies forward at its current depth.
            effective = min(self._cliff_top_target_z, vehicle_depth)
            for ix in range(self.cx, self.nx):
                self.cmd_depth[ix] = effective
            self._set_mode_from_cmd_depth(vehicle_depth)
            return

        # ----- Step 3: safety tail cap (always active) -----
        if self._safety_tail_blocked(vehicle_depth):
            for ix in range(self.cx, self.nx):
                if self.cmd_depth[ix] > vehicle_depth:
                    self.cmd_depth[ix] = vehicle_depth

        # ----- Mode selection -----
        self._set_mode_from_cmd_depth(vehicle_depth)

    def _set_mode_from_cmd_depth(self, vehicle_depth: float):
        """Pick mode from ``cmd_depth[cx] - vehicle_depth``."""
        c = self.cfg
        target_z = self.cmd_depth[self.cx]
        if np.isnan(target_z):
            self.control_mode = "ALT_FOLLOW"
            return
        dz = target_z - vehicle_depth
        if dz < -c.altitude_overshoot_threshold_m:
            self.control_mode = "OBSTACLE_CLEAR"
        elif dz > c.altitude_overshoot_threshold_m:
            self.control_mode = "ALT_CORRECTION"
        else:
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

    def update(self, vehicle_depth: float):
        """
        Run the full processing pipeline: manifold extraction, path planning,
        and waypoint generation. Call this after sensor updates and advance().

        Args:
            vehicle_depth: Current vehicle depth (m).

        Returns:
            Commanded depth at the vehicle position.
        """
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

        lines = [
            f"  mode={self.control_mode:<15}  "
            f"dvl_alt={self.dvl_altitude:.3f}m  target={c.imaging_altitude:.2f}m  "
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
