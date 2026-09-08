// ===========================================================================
// AUV 2D Obstacle Avoidance - Occupancy Map, Cliff Manifold, and Path Planner
//
// A vehicle-relative 2D occupancy grid (X forward, Z down) for seafloor imaging
// AUV obstacle avoidance. The map operates like a side-scrolling window centered
// on the vehicle, with a configurable forward and backward horizon.
//
// Sensor observations from a downward-facing DVL (3 Janus beams + altimeter)
// and a forward-looking sonar are projected into the 2D X-Z voxel space,
// ignoring lateral (Y) offsets.
//
// The "cliff world" manifold extracts the top surface of occupied voxels as a
// stair-step polyline.  A path planner generates a commanded depth profile over
// this manifold using several control modes with fixed priority:
//
//   1) Forward obstacle clearance (highest).  While the cliff-top latch is
//      active, OBSTACLE_CLEAR / OBSTACLE_HOLD dominate — ascent or forward
//      clear at latch depth until `release_x`.  Outside the latch,
//      OBSTACLE_CLEAR (vehicle below commanded depth threshold) similarly
//      takes precedence — no tail-only override.
//
//   2) Tail clearance vs altitude following / correction.  If the tail
//      safety band fires (see `safety_tail_blocked()`) and the vehicle would
//      otherwise be ALT_FOLLOW or ALT_CORRECTION, switch to TAIL_CLEAR constant-
//      depth forward flight until the tail clears.
//
//   3) Else ALT_FOLLOW (terrain-following / imaging_altitude),
//      or ALT_CORRECTION (descend in place when altitude is high).
//
// Individual modes:
//
//   Mode: ALT_FOLLOW  (default when forward + tail constraints satisfied)
//     Track imaging_altitude above the raw manifold (terrain-following).
//
//   Mode: OBSTACLE_CLEAR  (forward obstacle or commanded climb, vehicle deep)
//     Vehicle ascends in place (vx=0) toward the latch target depth, or toward
//     `cmd_depth[cx]` when no latch applies.
//
//   Mode: OBSTACLE_HOLD  (latch active, vehicle at or above target depth)
//     Vehicle flies forward at survey_speed holding the latch target depth
//     (DEPTH_HOLD).  Active until the vehicle passes release_x (target_x +
//     cliff_standoff + vehicle_length m).  The latch can only be updated
//     to a shallower target — never overridden or released early.
//
//   Mode: ALT_CORRECTION  (altitude diverges above imaging altitude target)
//     Triggered when vehicle depth is shallower than `cmd_depth[cx]` by more
//     than `altitude_overshoot_threshold_m`.  Vehicle stops forward motion
//     and descends in place toward imaging altitude unless TAIL_CLEAR applies.
//
//   Mode: TAIL_CLEAR  (tail clearance — overrides ALT_FOLLOW / ALT_CORRECTION)
//     Terrain within `safety_below_m` below the vehicle is detected in the
//     tail window (vehicle centre back to `safety_standoff_m` behind the
//     tail).  Vehicle drives forward at survey_speed at current depth
//     (DEPTH_HOLD) until `safety_tail_blocked()` clears, without diving for
//     imaging altitude.
//
// Coordinate conventions:
//     X: forward (positive ahead of vehicle)
//     Z: depth (positive downward)
//     Y: lateral (ignored in 2D projection)
//
// Usage — the grid directly:
//
//     #include "occupancy_map.hpp"
//     using namespace auv_oavoid;
//
//     OccupancyMap omap{OccupancyMapConfig{}};
//     omap.update_dvl_ray(ranges, beam_angles, vehicle_depth, vehicle_world_x);
//     omap.update_sonar(sonar_range, half_angle, vehicle_depth, world_x, hit);
//     omap.advance(ds);                  // as the vehicle moves forward
//     omap.update(vehicle_depth, vehicle_heading);
//     double cmd = omap.get_commanded_depth_at_vehicle();
//
// Usage — via ObstacleMapper, which owns the grid, applies pose bookkeeping
// and is what an integrating node should normally use:
//
//     ObstacleMapper mapper{cfg, dvl_cfg, sonar_cfg, alt_cfg};
//     mapper.update_sensor(SensorType::DVL, dvl_measurement, pose);
//     ControlCommand cmd = mapper.get_control();
//
// The same API is exposed to Python by the pybind11 module
// `occupancy_map_cpp` (see cpp/python/bindings.cpp), which the simulator and
// visualizers in this repo use.
//
// This C++ core is the single implementation of the algorithm.  A Python
// prototype previously ran alongside it; it was removed once the C++ was
// validated on the vehicle, and this file carries the design notes it held.
// ===========================================================================

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace auv_oavoid {

static constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// Sensor / mission configuration structs
// ---------------------------------------------------------------------------

struct DVLConfig {
    /// Each entry: {slant_angle_deg, heading_offset_deg}
    std::vector<std::pair<double, double>> beams = {
        {20.0,  0.0},
        {20.0, 120.0},
        {20.0, 240.0},
    };
    /// Simulator-only: the ray-caster needs a cutoff at which to stop marching
    /// and report a no-return.  A real DVL applies its own range limit and
    /// reports per-beam validity, so vehicle-side adapters must gate on those
    /// flags rather than on a configured maximum.
    double max_range = 50.0;

    /// 2-D projected angle from vertical for each beam (rad).
    /// atan2(sin(slant)*cos(heading_offset), cos(slant))
    std::vector<double> beam_angles_rad() const;

    /// Unit vectors per beam in vehicle frame (forward, starboard, down).
    /// Returns n_beams × 3 matrix.
    Eigen::MatrixXd beam_directions_3d() const;

    /// True for beams whose 3-D direction lies in the vehicle X-Z plane
    /// (no lateral/starboard displacement). Only these beams may clear voxels
    /// in the 2-D occupancy grid; sideways beams would incorrectly clear
    /// voxels they never actually passed through in 3-D space.
    std::vector<bool> beam_can_clear() const;
};

struct SonarConfig {
    double max_range  = 12.0;
    double half_angle = 3.0;   // half beam width (degrees)
    double noise_std  = 0.3;   // range measurement noise std (m)

    double half_angle_rad() const { return half_angle * M_PI / 180.0; }
};

struct AltimeterConfig {
    double max_range = 100.0;
};

struct Pose {
    double north;
    double east;
    double depth;
    double heading;
};

struct ControlCommand {
    double      vx;
    std::string vertical_mode;   // "ALT_FOLLOW" | "DEPTH_HOLD"
    double      vertical_target;
};

// ---------------------------------------------------------------------------
// OccupancyMapConfig — all 30+ fields, same names and defaults as Python
// ---------------------------------------------------------------------------

struct OccupancyMapConfig {
    // Voxel grid dimensions
    double dx            = 0.5;
    double dz            = 0.25;
    double horizon_fwd   = 15.0;
    double horizon_back  = 15.0;
    /// The grid spans [vehicle_z - z_half_range, vehicle_z + z_half_range]
    /// and shifts with the vehicle to keep it centred.
    double z_half_range  = 20.0;

    // Vehicle parameters
    double vehicle_length    = 2.0;
    double imaging_altitude  = 2.0;
    double survey_speed      = 0.5;
    double vertical_speed    = 0.5;

    // Path planning
    double cliff_standoff        = 2.0;
    double obstacle_threshold    = 1.0;

    double altitude_overshoot_threshold_m  = 1.0;
    /// Hysteresis on top of altitude_overshoot_threshold_m when choosing
    /// OBSTACLE_CLEAR vs ALT_FOLLOW vs ALT_CORRECTION.  Once in an in-place
    /// vertical transect mode, retain it until commanded depth agrees with the
    /// vehicle depth within roughly (threshold - hysteresis); damps chatter
    /// when noisy manifold/DVL jitter steps cmd_depth[cx] across the
    /// threshold (common on procedural / rugose reef terrain).
    double altitude_overshoot_hysteresis_m = 0.5;

    /// Stale-observation heading gate.  Any occupied voxel whose stored
    /// observation heading differs from the current vehicle heading by more
    /// than this threshold is reset to prior probability.  Prevents filled
    /// voxels from a previous heading from triggering obstacle-avoidance on a
    /// new heading where that space is actually clear.
    double stale_heading_threshold_deg = 45.0;

    // Safety clearance
    /// Minimum horizontal clearance behind the vehicle, from manifold geometry
    /// in the tail depth band, before descent is allowed.
    double safety_standoff_m = 2.0;
    /// Depth band used by the tail-clearance check: structure within
    /// [v_z, v_z + safety_below_m) below the vehicle counts as a tail threat.
    double safety_below_m    = 1.0;

    // Occupancy probability
    double prior       = 0.5;
    double occ_thresh  = 0.62;

    // DVL observation model
    double dvl_hit_prob   = 0.5;
    double dvl_miss_prob  = 0.3;
    double dvl_max_occ    = 0.98;
    double dvl_min_occ    = 0.02;

    // Altimeter observation model
    double altimeter_hit_prob  = 0.5;
    double altimeter_miss_prob = 0.3;
    double altimeter_max_occ   = 0.98;
    double altimeter_min_occ   = 0.02;

    // Forward sonar observation model
    double sonar_hit_prob  = 0.3;
    double sonar_miss_prob = 0.2;
    double sonar_max_occ   = 0.98;
    double sonar_min_occ   = 0.02;
    double sonar_min_depth_m = 1.0; ///< Ignore sonar returns when vehicle depth < this (surface reflection rejection)
};

// ---------------------------------------------------------------------------
// Measurement types
// ---------------------------------------------------------------------------

enum class SensorType { DVL, ALTIMETER, SONAR };

struct DVLMeasurement {
    std::vector<double> ranges;
    std::vector<bool>   hit_surface;
};

struct AltimeterMeasurement {
    double range_m;
    bool   hit = true;
};

struct SonarMeasurement {
    double range_m;
    bool   hit;
};

// ---------------------------------------------------------------------------
// OccupancyMap
// ---------------------------------------------------------------------------

class OccupancyMap {
public:
    explicit OccupancyMap(const OccupancyMapConfig& cfg = {});

    void reset(double vehicle_world_x = 0.0, double vehicle_depth = 0.0);

    // ---- Coordinate transforms ----
    std::pair<int, int> world_to_grid(double wx, double wz) const;
    double grid_to_world_x(int ix) const;
    double grid_to_world_z(int iz) const;

    // ---- Grid management ----
    void advance(double dx);
    void shift_depth(double vehicle_z);

    // ---- Sensor updates ----
    /// Update occupancy by ray-marching each DVL beam, and track direct altitude.
    ///
    /// Marks cells along the beam as free, and the endpoint cell as occupied.
    /// This provides more information per observation than endpoint-only updates.
    ///
    /// Also computes self.dvl_altitude — the minimum vertical-axis range
    /// component across all beams that actually hit the seafloor:
    ///
    ///     dvl_altitude = min( range_i * cos(angle_i) )   for hit beams
    ///
    /// If no beam records a valid bottom return, `self.dvl_altitude` is set
    /// to `NaN` so planners do not reuse a stale altitude from a previous
    /// cycle.  Beam validity comes entirely from `hit_surface` — the caller's
    /// adapter is responsible for translating its sensor's own no-return
    /// convention into that flag.
    ///
    /// This is the shortest terrain clearance observed by any beam, measured
    /// along the depth axis.  The altimeter (angle=0) contributes directly;
    /// Janus beams at ±25° contribute via their cos(25°) ≈ 0.906 factor.
    /// On rising terrain the forward Janus beam may detect shallower ground
    /// before the altimeter, giving earlier terrain-following response.
    ///
    /// dvl_altitude is updated only when hit_surface is provided.  Pass
    /// hit_surface=None to perform occupancy updates without touching the
    /// altitude estimate (e.g. when hit flags are unavailable).
    ///
    /// Args:
    ///     ranges: Range measurements per beam (m). Shape (n_beams,).
    ///     beam_angles: Beam angles from vertical (rad). Shape (n_beams,).
    ///     vehicle_depth: Current vehicle depth (m).
    ///     vehicle_world_x: Current vehicle world X position (m).
    ///     hit_surface: Boolean array, True if beam hit the seafloor.
    ///                  Shape (n_beams,).  If None, altitude is not updated.
    ///     range_step: Step size for ray marching (m).
    void update_dvl_ray(
        const std::vector<double>& ranges,
        const std::vector<double>& beam_angles,
        double vehicle_depth,
        double vehicle_world_x,
        const std::optional<std::vector<bool>>& hit_surface = std::nullopt,
        double range_step    = 0.15,
        double vehicle_heading = kNaN,
        const std::optional<std::vector<bool>>& can_clear = std::nullopt);

    /// Update occupancy from a straight-down altimeter beam.
    ///
    /// Ray-marches vertically from vehicle depth to range_m, marking cells
    /// free along the path and occupied (or free on miss) at the endpoint.
    ///
    /// Args:
    ///     range_m:          Measured range to seafloor (m).
    ///     vehicle_depth:    Current vehicle depth (m).
    ///     vehicle_world_x:  Vehicle X position in world frame (m).
    ///     hit:              True if the beam returned a valid seafloor return.
    ///     range_step:       Ray-march step size (m).
    ///     vehicle_heading:  Vehicle heading (rad) for voxel metadata.
    void update_altimeter_ray(
        double range_m,
        double vehicle_depth,
        double vehicle_world_x,
        bool   hit,
        double range_step    = 0.15,
        double vehicle_heading = kNaN);

    /// Update occupancy from forward-looking sonar observation.
    ///
    /// The sonar cone is projected into the X-Z plane. Uses lower confidence
    /// updates than DVL to avoid false positive obstacle detections.
    ///
    /// Args:
    ///     sonar_range: Measured range (m), or max range if no return.
    ///     sonar_half_angle: Half-angle of sonar beam (rad).
    ///     vehicle_depth: Current vehicle depth (m).
    ///     vehicle_world_x: Current vehicle world X position (m).
    ///     hit_obstacle: True if sonar detected a return.
    ///     range_step: Step size for ray marching (m).
    ///     angle_steps: Number of angular samples across the beam.
    void update_sonar(
        double sonar_range,
        double sonar_half_angle,
        double vehicle_depth,
        double vehicle_world_x,
        bool   hit_obstacle,
        double range_step    = 0.2,
        int    angle_steps   = 7,
        double vehicle_heading = kNaN);

    // ---- Planning pipeline ----
    /// Extract the cliff manifold from the occupancy grid.
    ///
    /// For each X column, finds the shallowest (topmost) occupied voxel.  The
    /// manifold runs along the top of occupied space as a stair-step polyline
    /// — it always goes over obstacles, never underneath.
    ///
    /// Every column always has a manifold value.  Unobserved columns default
    /// to the grid bottom depth ("we don't know — assume terrain is at the
    /// depth limit of the occupancy map").  Direct observations override the
    /// default where present.  AHEAD of the vehicle, forward-extension fills
    /// still-unobserved columns past the last ahead observation with the
    /// last observed depth (flat extension assumption).  BEHIND the vehicle
    /// there is NO backward forward-extension: a single backward beam hit
    /// does not propagate into adjacent never-observed columns.
    ///
    /// Updates manifold_iz_ and manifold_z_ in place.
    void   build_cliff_manifold();

    /// Generate a commanded depth profile over the occupancy grid.
    ///
    /// Pipeline:
    ///
    ///   Step 1 — Altitude-following baseline + DVL override at vehicle column.
    ///     `cmd_depth[ix] = manifold[ix] - imaging_altitude`.  At the
    ///     vehicle column `cx`, override with the min-across-beams DVL
    ///     altitude reading so the altimeter + forward Janus give earliest
    ///     terrain response.
    ///
    ///   Step 2 — Cliff-top latch.
    ///     Approach: climb_target caps cmd_depth at `peak_z - imaging`
    ///     while the vehicle climbs to maintain `imaging_altitude` distance
    ///     from upcoming terrain.
    ///     Commit: when the vehicle has crossed over the highest detected
    ///     voxel within `cliff_standoff`, latch on.  Hold target depth
    ///     (= peak_z - imaging) and fly forward
    ///     `cliff_standoff + vehicle_length` m at constant depth so
    ///     the tail clears the cliff edge.  During the forward-hold, if any
    ///     DVL beam reads altitude < `imaging_altitude`, ratchet target
    ///     shallower to maintain that minimum (vehicle never descends in
    ///     this phase).  Release after the forward-hold distance.
    ///
    ///   Step 3 — Safety tail cap (always active).
    ///     Caps cmd_depth at `vehicle_depth` when manifold is within
    ///     `safety_below_m` below the vehicle AND within
    ///     `safety_standoff_m` horizontal of the tail.
    ///
    /// Mode selection from final `cmd_depth[cx] - vehicle_depth` (when the
    /// cliff latch does not consume the cycle), using a Schmitt-like band via
    /// `altitude_overshoot_hysteresis_m` so small `cmd_depth` jitter does not
    /// flip OBSTACLE_CLEAR ↔ ALT_CORRECTION:
    ///   OBSTACLE_CLEAR — vehicle below target (must ascend in place).
    ///   ALT_CORRECTION — vehicle above target (must descend in place).
    ///   ALT_FOLLOW    — within hysteresis-augmented band of target.
    /// Then, if `_safety_tail_blocked` and mode is ALT_FOLLOW or
    /// ALT_CORRECTION → TAIL_CLEAR (constant-depth forward flight).  Cliff
    /// latch paths never reach this hook — forward obstacle dominates.
    ///
    /// Args:
    ///     vehicle_depth: Current vehicle depth (m).
    void   build_commanded_depth(double vehicle_depth);

    void   build_path_waypoints();
    double get_commanded_depth_at_vehicle() const;

    /// Reset any observed voxel whose observation heading is too far from current heading.
    ///
    /// `voxel_heading` records the vehicle heading at the time of the most
    /// recent observation of each voxel, whether that observation was a hit
    /// (occupied) or a miss (free).  Any voxel whose stored heading differs
    /// from *vehicle_heading* by more than `stale_heading_threshold_deg` is
    /// reset to prior (unobserved) probability and its heading cleared.
    ///
    /// Exception: occupied voxels in the columns directly under the vehicle
    /// footprint have their stored heading refreshed to the current heading
    /// instead of being cleared.  These are real terrain observations the
    /// vehicle is flying over — discarding them on a heading change would
    /// cause the planner to lose the seafloor directly below.  Free voxels
    /// under the footprint are still cleared when stale, since a free-space
    /// observation from one heading may not hold from another.
    ///
    /// Args:
    ///     vehicle_heading: Current vehicle heading (radians, same convention
    ///                      as headings stored by the sensor update methods).
    void   clear_stale_voxels(double vehicle_heading);

    /// Run the full processing pipeline: stale-voxel clearing, manifold
    /// extraction, path planning, and waypoint generation. Call this after
    /// sensor updates.
    ///
    /// Args:
    ///     vehicle_depth:   Current vehicle depth (m).
    ///     vehicle_heading: Current vehicle heading (radians).  When provided,
    ///                      occupied voxels whose observation heading differs by
    ///                      more than `stale_heading_threshold_deg` are reset
    ///                      to prior before planning.
    ///
    /// Returns:
    ///     Commanded depth at the vehicle position.
    double update(double vehicle_depth, double vehicle_heading = kNaN);

    // ---- Accessors ----
    OccupancyMapConfig&       cfg()       { return cfg_; }
    const OccupancyMapConfig& cfg() const { return cfg_; }

    int    nx() const { return nx_; }
    int    nz() const { return nz_; }
    int    cx() const { return cx_; }

    double grid_origin_x()       const { return grid_origin_x_; }
    double grid_origin_z()       const { return grid_origin_z_; }
    double manifold_grid_origin_x() const { return manifold_grid_origin_x_; }
    double dvl_altitude()        const { return dvl_altitude_; }

    /// Latest valid altimeter vertical range, or NaN after a no-return.
    /// Recorded here (as well as on ObstacleMapper) because the planner needs a
    /// direct altitude measurement that survives a heading change wiping the
    /// map.  ObstacleMapper keeps the two in step.
    double altimeter_altitude()  const { return altimeter_altitude_; }
    void   set_altimeter_altitude(double v) { altimeter_altitude_ = v; }

    /// Overwrite the cached DVL altitude.  Intended for modelling a sensor
    /// dropout — the map normally maintains this itself from beam returns, and
    /// NaNs it when no beam reports a bottom lock.
    void   set_dvl_altitude(double v)  { dvl_altitude_ = v; }
    double shift_accum()         const { return shift_accum_; }

    const std::string&                    control_mode()      const { return control_mode_; }
    const Eigen::MatrixXd&                grid()              const { return grid_; }
    const Eigen::VectorXd&                manifold_z()        const { return manifold_z_; }
    const Eigen::VectorXi&                manifold_iz()       const { return manifold_iz_; }
    const std::vector<bool>&              manifold_observed() const { return manifold_observed_; }
    const Eigen::VectorXd&                cmd_depth()         const { return cmd_depth_; }
    const std::vector<std::pair<double,double>>& path_waypoints() const { return path_waypoints_; }

    std::string get_debug_summary(double vehicle_depth) const;

private:
    OccupancyMapConfig cfg_;
    int nx_, nz_, cx_;

    Eigen::MatrixXd grid_;          // (nz, nx) — grid(iz, ix)
    Eigen::MatrixXd voxel_heading_; // (nz, nx) — NaN for unobserved

    double grid_origin_x_;
    /// Grid origin in world Z (depth) — row 0 corresponds to this depth.
    /// Shifts with the vehicle so the vehicle stays centred in the window.
    double grid_origin_z_;
    double shift_accum_;
    double shift_accum_z_;

    Eigen::VectorXi manifold_iz_;       // length nx
    Eigen::VectorXd manifold_z_;        // length nx
    /// Parallel "is this column based on a real observation" flag.  False
    /// where manifold_z_ is the grid-bottom default — those columns are truly
    /// unknown, and the planner's obstacle-avoidance checks (climb, cliff
    /// drop, tail clearance) must skip transitions involving them to avoid
    /// false triggers from observed->default discontinuities.  True for direct
    /// observations AND forward-extended ahead columns (both reflect real
    /// terrain knowledge).
    std::vector<bool> manifold_observed_; // length nx
    Eigen::VectorXd cmd_depth_;         // length nx

    /// Grid origin that was in effect when manifold_z_ was last computed.
    /// Kept separately from grid_origin_x_ so a visualizer can map
    /// manifold_z_[i] -> world_x = manifold_grid_origin_x_ + i * dx even after
    /// advance() has shifted grid_origin_x_ forward by one column.
    double manifold_grid_origin_x_;
    double dvl_altitude_;
    double altimeter_altitude_;
    std::string control_mode_;

    std::vector<std::pair<double, double>> path_waypoints_;

    // Cliff-top latch state
    bool   cliff_top_committed_;
    double cliff_top_target_z_;
    double cliff_top_target_x_;
    double cliff_top_release_x_;
    double cliff_top_commit_heading_;  // heading when the latch anchored to its peak

    double last_vehicle_heading_;      // set by update()

    // ---- Internal helpers ----
    bool in_bounds(int ix, int iz) const {
        return ix >= 0 && ix < nx_ && iz >= 0 && iz < nz_;
    }

    struct ObstacleResult {
        bool   found;
        double peak_z;
        double peak_world_x;
    };

    ObstacleResult forward_obstacle(
        double vehicle_depth,
        double vehicle_world_x,
        double z_threshold = kNaN) const;

    bool safety_tail_blocked(double vehicle_depth) const;

    void set_mode_from_cmd_depth(double vehicle_depth);
};

// ---------------------------------------------------------------------------
// ObstacleMapper — thread-safe high-level interface
// ---------------------------------------------------------------------------

/// Thread-safe AUV obstacle avoidance interface.
///
/// Accepts asynchronous sensor measurements from DVL, altimeter, and forward
/// sonar via update_sensor(). Advances the occupancy grid based on the forward
/// displacement between consecutive poses, accounting for vehicle heading.
///
/// Velocity commands suitable for a real AUV controller are returned by
/// get_control(). Vehicle altitude (minimum of last DVL and altimeter vertical
/// readings) is returned by get_altitude().
///
/// The forward sonar origin is at the vehicle nose (vehicle_center +
/// vehicle_length/2). Sonar ranges are measured from that point.
///
/// Args:
///     config:            Occupancy map and path-planning parameters.
///     dvl_config:        DVL beam geometry and max range.
///     sonar_config:      Forward sonar beam width and max range.
///     altimeter_config:  Downward altimeter max range (optional).
class ObstacleMapper {
public:
    ObstacleMapper(
        const OccupancyMapConfig& cfg,
        const DVLConfig&          dvl_config,
        const SonarConfig&        sonar_config,
        const AltimeterConfig&    altimeter_config = {});

    void reset(const Pose& pose);

    // Three overloads matching sensor type
    void update_sensor(SensorType type, const DVLMeasurement&,       const Pose&);
    void update_sensor(SensorType type, const AltimeterMeasurement&, const Pose&);
    void update_sensor(SensorType type, const SonarMeasurement&,     const Pose&);

    void update_pose(const Pose& pose);

    /// Overwrite the cached altimeter altitude.  As with
    /// OccupancyMap::set_dvl_altitude, this exists to model a sensor dropout;
    /// normal operation maintains it from altimeter measurements.  Clears the
    /// map's copy too, which the planner reads.
    void set_altimeter_altitude(double v) {
        altimeter_altitude_ = v;
        omap_.set_altimeter_altitude(v);
    }

    /// Return the current obstacle avoidance command.
    ///
    /// Priority: forward obstacle (OBSTACLE_CLEAR / OBSTACLE_HOLD from latch or
    /// below-target ascent) unchanged; tail blocked with ALT_FOLLOW or
    /// ALT_CORRECTION → TAIL_CLEAR (survey_speed, DEPTH_HOLD at present depth).
    ///
    /// ALT_FOLLOW / ALT_CORRECTION otherwise:
    ///     vertical_mode  = 'ALT_FOLLOW'
    ///     vertical_target = imaging_altitude (m above seafloor)
    ///     vx             = survey_speed or 0 (ALT_CORRECTION)
    ///
    /// OBSTACLE_CLEAR: DEPTH_HOLD at cmd depth, vx=0 until within band.
    ///
    /// OBSTACLE_HOLD / TAIL_CLEAR: DEPTH_HOLD, vx=survey_speed.
    ///
    /// Thread-safe.
    ControlCommand get_control();
    double         get_altitude();

    OccupancyMap&       omap()       { return omap_; }
    const OccupancyMap& omap() const { return omap_; }

private:
    OccupancyMap    omap_;
    DVLConfig       dvl_config_;
    SonarConfig     sonar_config_;
    AltimeterConfig altimeter_config_;
    std::mutex      lock_;

    std::optional<Pose> last_pose_;
    double              altimeter_altitude_;

    void   advance_to_pose(const Pose& pose);
    double vehicle_forward_x() const;
};

} // namespace auv_oavoid
