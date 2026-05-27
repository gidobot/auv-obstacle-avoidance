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
        {0.0,   0.0},
        {20.0,  0.0},
        {20.0, 120.0},
        {20.0, 240.0},
    };
    double max_range = 50.0;

    /// 2-D projected angle from vertical for each beam (rad).
    /// atan2(sin(slant)*cos(heading_offset), cos(slant))
    std::vector<double> beam_angles_rad() const;

    /// Unit vectors per beam in vehicle frame (forward, starboard, down).
    /// Returns n_beams × 3 matrix.
    Eigen::MatrixXd beam_directions_3d() const;
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
    double altitude_overshoot_hysteresis_m = 0.5;

    // Stale-observation heading gate
    double stale_heading_threshold_deg = 45.0;

    // Safety clearance
    double safety_standoff_m = 2.0;
    double safety_below_m    = 1.0;

    // Occupancy probability
    double prior       = 0.5;
    double occ_thresh  = 0.62;

    // DVL observation model
    double dvl_hit_prob   = 0.5;
    double dvl_miss_prob  = 0.3;
    double dvl_max_occ    = 0.98;
    double dvl_min_occ    = 0.02;
    double dvl_max_range_m = 50.0;

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
    void update_dvl_ray(
        const std::vector<double>& ranges,
        const std::vector<double>& beam_angles,
        double vehicle_depth,
        double vehicle_world_x,
        const std::optional<std::vector<bool>>& hit_surface = std::nullopt,
        double range_step    = 0.15,
        double vehicle_heading = kNaN);

    void update_altimeter_ray(
        double range_m,
        double vehicle_depth,
        double vehicle_world_x,
        bool   hit,
        double range_step    = 0.15,
        double vehicle_heading = kNaN);

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
    void   build_cliff_manifold();
    void   build_commanded_depth(double vehicle_depth);
    void   build_path_waypoints();
    double get_commanded_depth_at_vehicle() const;
    void   clear_stale_voxels(double vehicle_heading);
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
    double grid_origin_z_;
    double shift_accum_;
    double shift_accum_z_;

    Eigen::VectorXi manifold_iz_;       // length nx
    Eigen::VectorXd manifold_z_;        // length nx
    std::vector<bool> manifold_observed_; // length nx
    Eigen::VectorXd cmd_depth_;         // length nx

    double manifold_grid_origin_x_;
    double dvl_altitude_;
    std::string control_mode_;

    std::vector<std::pair<double, double>> path_waypoints_;

    // Cliff-top latch state
    bool   cliff_top_committed_;
    double cliff_top_target_z_;
    double cliff_top_target_x_;
    double cliff_top_release_x_;

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
