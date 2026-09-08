#include "occupancy_map.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <sstream>
#include <iomanip>

namespace auv_oavoid {

// ===========================================================================
// DVLConfig helpers
// ===========================================================================

std::vector<double> DVLConfig::beam_angles_rad() const {
    std::vector<double> angles;
    angles.reserve(beams.size());
    for (auto& [slant_deg, h_off_deg] : beams) {
        double s = slant_deg * M_PI / 180.0;
        double h = h_off_deg * M_PI / 180.0;
        angles.push_back(std::atan2(std::sin(s) * std::cos(h), std::cos(s)));
    }
    return angles;
}

Eigen::MatrixXd DVLConfig::beam_directions_3d() const {
    int n = static_cast<int>(beams.size());
    Eigen::MatrixXd dirs(n, 3);
    for (int i = 0; i < n; ++i) {
        double s = beams[i].first  * M_PI / 180.0;
        double h = beams[i].second * M_PI / 180.0;
        dirs(i, 0) = std::sin(s) * std::cos(h); // forward
        dirs(i, 1) = std::sin(s) * std::sin(h); // starboard
        dirs(i, 2) = std::cos(s);               // down
    }
    return dirs;
}

std::vector<bool> DVLConfig::beam_can_clear() const {
    std::vector<bool> result;
    result.reserve(beams.size());
    for (auto& [slant_deg, h_off_deg] : beams) {
        double s = slant_deg  * M_PI / 180.0;
        double h = h_off_deg  * M_PI / 180.0;
        double lateral = std::sin(s) * std::sin(h); // starboard component
        result.push_back(std::abs(lateral) < 1e-9);
    }
    return result;
}

// ===========================================================================
// OccupancyMap — construction & reset
// ===========================================================================

OccupancyMap::OccupancyMap(const OccupancyMapConfig& cfg)
    : cfg_(cfg)
{
    if (cfg_.dx <= 0.0 || cfg_.dz <= 0.0)
        throw std::invalid_argument("OccupancyMapConfig: dx and dz must be positive");

    nx_ = static_cast<int>(std::ceil((cfg_.horizon_fwd + cfg_.horizon_back) / cfg_.dx));
    nz_ = static_cast<int>(std::ceil(2.0 * cfg_.z_half_range / cfg_.dz));
    cx_ = static_cast<int>(std::floor(cfg_.horizon_back / cfg_.dx));

    grid_          = Eigen::MatrixXd::Constant(nz_, nx_, cfg_.prior);
    voxel_heading_ = Eigen::MatrixXd::Constant(nz_, nx_, kNaN);

    grid_origin_x_ = 0.0;
    grid_origin_z_ = -(nz_ / 2) * cfg_.dz;

    shift_accum_   = 0.0;
    shift_accum_z_ = 0.0;

    manifold_iz_       = Eigen::VectorXi::Constant(nx_, nz_ - 1);
    manifold_z_        = Eigen::VectorXd::Constant(nx_, grid_origin_z_ + (nz_ - 1) * cfg_.dz);
    manifold_observed_ = std::vector<bool>(nx_, false);
    cmd_depth_         = Eigen::VectorXd::Constant(nx_, kNaN);

    manifold_grid_origin_x_ = 0.0;
    dvl_altitude_           = kNaN;
    altimeter_altitude_     = kNaN;
    control_mode_           = "ALT_FOLLOW";
    path_waypoints_         = {};

    cliff_top_committed_ = false;
    cliff_top_target_z_  = kNaN;
    cliff_top_target_x_  = kNaN;
    cliff_top_release_x_ = kNaN;
    cliff_top_commit_heading_ = kNaN;
    last_vehicle_heading_     = kNaN;
}

void OccupancyMap::reset(double vehicle_world_x, double vehicle_depth) {
    grid_.setConstant(cfg_.prior);
    voxel_heading_.setConstant(kNaN);

    grid_origin_x_          = vehicle_world_x - cx_ * cfg_.dx;
    grid_origin_z_          = vehicle_depth - (nz_ / 2) * cfg_.dz;
    manifold_grid_origin_x_ = grid_origin_x_;

    shift_accum_   = 0.0;
    shift_accum_z_ = 0.0;

    manifold_iz_.setConstant(nz_ - 1);
    manifold_z_.setConstant(grid_origin_z_ + (nz_ - 1) * cfg_.dz);
    manifold_observed_.assign(nx_, false);
    cmd_depth_.setConstant(vehicle_depth);
    path_waypoints_.clear();

    dvl_altitude_  = kNaN;
    altimeter_altitude_ = kNaN;
    control_mode_  = "ALT_FOLLOW";

    cliff_top_committed_ = false;
    cliff_top_target_z_  = kNaN;
    cliff_top_target_x_  = kNaN;
    cliff_top_release_x_ = kNaN;
    cliff_top_commit_heading_ = kNaN;
    last_vehicle_heading_     = kNaN;
}

// ===========================================================================
// Coordinate transforms
// ===========================================================================

std::pair<int, int> OccupancyMap::world_to_grid(double wx, double wz) const {
    int ix = static_cast<int>(std::round((wx - grid_origin_x_) / cfg_.dx));
    int iz = static_cast<int>(std::round((wz - grid_origin_z_) / cfg_.dz));
    return {ix, iz};
}

double OccupancyMap::grid_to_world_x(int ix) const {
    return grid_origin_x_ + ix * cfg_.dx;
}

double OccupancyMap::grid_to_world_z(int iz) const {
    return grid_origin_z_ + iz * cfg_.dz;
}

// ===========================================================================
// Grid management
// ===========================================================================

void OccupancyMap::advance(double dx_dist) {
    shift_accum_ += dx_dist;
    int cols = static_cast<int>(std::floor(shift_accum_ / cfg_.dx));
    if (cols <= 0) return;

    shift_accum_   -= cols * cfg_.dx;
    grid_origin_x_ += cols * cfg_.dx;

    if (cols >= nx_) {
        grid_.setConstant(cfg_.prior);
        voxel_heading_.setConstant(kNaN);
    } else {
        // Shift columns left: new columns enter on the right.
        // grid_ is (nz, nx) — columns are major dimension in Eigen column-major.
        // grid_.leftCols(nx-cols) = grid_.rightCols(nx-cols)  — needs .eval() to avoid aliasing
        grid_.leftCols(nx_ - cols) = grid_.rightCols(nx_ - cols).eval();
        grid_.rightCols(cols).setConstant(cfg_.prior);

        voxel_heading_.leftCols(nx_ - cols) = voxel_heading_.rightCols(nx_ - cols).eval();
        voxel_heading_.rightCols(cols).setConstant(kNaN);
    }
}

void OccupancyMap::shift_depth(double vehicle_z) {
    double center_iz  = static_cast<double>(nz_ / 2);
    double vehicle_iz = (vehicle_z - grid_origin_z_) / cfg_.dz;
    shift_accum_z_ += vehicle_iz - center_iz;

    int rows = static_cast<int>(std::floor(std::abs(shift_accum_z_)));
    if (rows == 0) return;

    if (shift_accum_z_ < 0) rows = -rows;
    shift_accum_z_ -= rows;
    grid_origin_z_  += rows * cfg_.dz;

    if (rows > 0) {
        // Vehicle moved deeper: drop shallow rows, expose new deep rows at bottom.
        if (rows >= nz_) {
            grid_.setConstant(cfg_.prior);
            voxel_heading_.setConstant(kNaN);
        } else {
            // grid[:-rows, :] = grid[rows:, :]   (top rows become shallow, new deep exposed)
            // In Eigen (nz, nx): topRows(nz-rows) = bottomRows(nz-rows).eval()
            grid_.topRows(nz_ - rows) = grid_.bottomRows(nz_ - rows).eval();
            grid_.bottomRows(rows).setConstant(cfg_.prior);

            voxel_heading_.topRows(nz_ - rows) = voxel_heading_.bottomRows(nz_ - rows).eval();
            voxel_heading_.bottomRows(rows).setConstant(kNaN);
        }
    } else {
        // Vehicle moved shallower: drop deep rows, expose new shallow rows at top.
        int r = -rows;
        if (r >= nz_) {
            grid_.setConstant(cfg_.prior);
            voxel_heading_.setConstant(kNaN);
        } else {
            // grid[r:, :] = grid[:-r, :]
            // In Eigen: bottomRows(nz-r) = topRows(nz-r).eval()
            grid_.bottomRows(nz_ - r) = grid_.topRows(nz_ - r).eval();
            grid_.topRows(r).setConstant(cfg_.prior);

            voxel_heading_.bottomRows(nz_ - r) = voxel_heading_.topRows(nz_ - r).eval();
            voxel_heading_.topRows(r).setConstant(kNaN);
        }
    }
}

// ===========================================================================
// Sensor updates
// ===========================================================================

void OccupancyMap::update_dvl_ray(
    const std::vector<double>& ranges,
    const std::vector<double>& beam_angles,
    double vehicle_depth,
    double vehicle_world_x,
    const std::optional<std::vector<bool>>& hit_surface,
    double range_step,
    double vehicle_heading,
    const std::optional<std::vector<bool>>& can_clear)
{
    if (beam_angles.size() != ranges.size())
        throw std::invalid_argument("update_dvl_ray: beam_angles and ranges must be the same length");
    if (hit_surface.has_value() && hit_surface->size() != ranges.size())
        throw std::invalid_argument("update_dvl_ray: hit_surface must be the same length as ranges");
    if (can_clear.has_value() && can_clear->size() != ranges.size())
        throw std::invalid_argument("update_dvl_ray: can_clear must be the same length as ranges");

    const auto& c = cfg_;
    int n = static_cast<int>(ranges.size());

    for (int i = 0; i < n; ++i) {
        double r_max     = ranges[i];
        if (!std::isfinite(r_max)) continue;
        double ang       = beam_angles[i];
        bool   is_hit    = !hit_surface.has_value() || (*hit_surface)[i];
        bool   allow_clear = !can_clear.has_value() || (*can_clear)[i];

        // Ray-march free cells — axis-aligned beams only.
        // Lateral beams travel through different 3-D voxels than their 2-D
        // projection implies; clearing along their projected path would
        // incorrectly free voxels the beam never actually passed through.
        if (allow_clear) {
            for (double r = range_step; r < r_max - range_step; r += range_step) {
                double dx = std::sin(ang) * r;
                double dz = std::cos(ang) * r;
                auto [ix, iz] = world_to_grid(vehicle_world_x + dx, vehicle_depth + dz);
                if (!in_bounds(ix, iz)) continue;
                grid_(iz, ix) = std::max(c.dvl_min_occ, grid_(iz, ix) - c.dvl_miss_prob);
                if (grid_(iz, ix) <= c.occ_thresh) {
                    voxel_heading_(iz, ix) = vehicle_heading;
                }
            }
        }

        // Endpoint: always mark hit occupied; only clear on miss if axis-aligned.
        double dx = std::sin(ang) * r_max;
        double dz = std::cos(ang) * r_max;
        auto [ix, iz] = world_to_grid(vehicle_world_x + dx, vehicle_depth + dz);
        if (!in_bounds(ix, iz)) continue;

        if (is_hit) {
            grid_(iz, ix) = std::min(c.dvl_max_occ, grid_(iz, ix) + c.dvl_hit_prob);
            voxel_heading_(iz, ix) = vehicle_heading;
        } else if (allow_clear) {
            grid_(iz, ix) = std::max(c.dvl_min_occ, grid_(iz, ix) - c.dvl_miss_prob);
            voxel_heading_(iz, ix) = vehicle_heading;
        }
    }

    // Update direct altitude estimate from surface-hitting beams.
    if (hit_surface.has_value()) {
        double min_vert = std::numeric_limits<double>::infinity();
        for (int i = 0; i < n; ++i) {
            if ((*hit_surface)[i]) {
                double vert = ranges[i] * std::cos(beam_angles[i]);
                if (vert < min_vert) min_vert = vert;
            }
        }
        dvl_altitude_ = std::isinf(min_vert) ? kNaN : min_vert;
    }
}

void OccupancyMap::update_altimeter_ray(
    double range_m,
    double vehicle_depth,
    double vehicle_world_x,
    bool   hit,
    double range_step,
    double vehicle_heading)
{
    if (!std::isfinite(range_m)) return;
    const auto& c = cfg_;

    // Ray-march free cells vertically
    for (double r = range_step; r < range_m - range_step; r += range_step) {
        auto [ix, iz] = world_to_grid(vehicle_world_x, vehicle_depth + r);
        if (!in_bounds(ix, iz)) continue;
        grid_(iz, ix) = std::max(c.altimeter_min_occ, grid_(iz, ix) - c.altimeter_miss_prob);
        if (grid_(iz, ix) <= c.occ_thresh) {
            voxel_heading_(iz, ix) = vehicle_heading;
        }
    }

    // Endpoint
    auto [ix, iz] = world_to_grid(vehicle_world_x, vehicle_depth + range_m);
    if (!in_bounds(ix, iz)) return;

    if (hit) {
        grid_(iz, ix) = std::min(c.altimeter_max_occ, grid_(iz, ix) + c.altimeter_hit_prob);
        voxel_heading_(iz, ix) = vehicle_heading;
    } else {
        grid_(iz, ix) = std::max(c.altimeter_min_occ, grid_(iz, ix) - c.altimeter_miss_prob);
        voxel_heading_(iz, ix) = vehicle_heading;
    }

    // Keep the latest direct vertical range for the planner.  A no-return
    // clears it rather than leaving a stale value in place.
    altimeter_altitude_ = hit ? range_m : kNaN;
}

void OccupancyMap::update_sonar(
    double sonar_range,
    double sonar_half_angle,
    double vehicle_depth,
    double vehicle_world_x,
    bool   hit_obstacle,
    double range_step,
    int    angle_steps,
    double vehicle_heading)
{
    if (!std::isfinite(sonar_range)) return;
    const auto& c = cfg_;
    if (vehicle_depth < c.sonar_min_depth_m) return;

    // Generate angular samples across beam (matches np.linspace)
    std::vector<double> angles(angle_steps);
    if (angle_steps == 1) {
        angles[0] = 0.0;
    } else {
        for (int k = 0; k < angle_steps; ++k) {
            angles[k] = -sonar_half_angle
                      + k * (2.0 * sonar_half_angle) / (angle_steps - 1);
        }
    }

    for (double ang : angles) {
        // Ray-march free cells
        for (double r = range_step; r < sonar_range - range_step; r += range_step) {
            double dx = r * std::cos(ang);
            double dz = r * std::sin(ang);
            auto [ix, iz] = world_to_grid(vehicle_world_x + dx, vehicle_depth + dz);
            if (!in_bounds(ix, iz)) continue;
            grid_(iz, ix) = std::max(c.sonar_min_occ, grid_(iz, ix) - c.sonar_miss_prob);
            if (grid_(iz, ix) <= c.occ_thresh) {
                voxel_heading_(iz, ix) = vehicle_heading;
            }
        }

        // Endpoint
        double dx = sonar_range * std::cos(ang);
        double dz = sonar_range * std::sin(ang);
        auto [ix, iz] = world_to_grid(vehicle_world_x + dx, vehicle_depth + dz);
        if (!in_bounds(ix, iz)) continue;

        if (hit_obstacle) {
            grid_(iz, ix) = std::min(c.sonar_max_occ, grid_(iz, ix) + c.sonar_hit_prob);
            voxel_heading_(iz, ix) = vehicle_heading;
        } else {
            grid_(iz, ix) = std::max(c.sonar_min_occ, grid_(iz, ix) - c.sonar_miss_prob);
            voxel_heading_(iz, ix) = vehicle_heading;
        }
    }
}

// ===========================================================================
// Cliff manifold extraction
// ===========================================================================

void OccupancyMap::build_cliff_manifold() {
    const auto& c = cfg_;
    // Snapshot the origin now so a visualizer can correctly map
    // manifold_z_[i] -> world_x after advance() has shifted grid_origin_x_.
    manifold_grid_origin_x_ = grid_origin_x_;

    int    bottom_iz = nz_ - 1;
    double bottom_z  = grid_to_world_z(bottom_iz);

    // Pass 1: shallowest occupied voxel per column (-1 if none).
    std::vector<int> observed_iz(nx_, -1);
    for (int ix = 0; ix < nx_; ++ix) {
        for (int iz = 0; iz < nz_; ++iz) {
            if (grid_(iz, ix) > c.occ_thresh) {
                observed_iz[ix] = iz;
                break;
            }
        }
    }

    // Pass 2 — behind the vehicle: observation OR grid-bottom default.
    for (int ix = 0; ix < cx_; ++ix) {
        if (observed_iz[ix] >= 0) {
            manifold_iz_[ix]       = observed_iz[ix];
            manifold_z_[ix]        = grid_to_world_z(observed_iz[ix]);
            manifold_observed_[ix] = true;
        } else {
            manifold_iz_[ix]       = bottom_iz;
            manifold_z_[ix]        = bottom_z;
            manifold_observed_[ix] = false;
        }
    }

    // Pass 3 — at and ahead of the vehicle: observation, else forward-extend
    // from the last ahead observation, else grid-bottom default.
    int last_iz = -1;
    for (int ix = cx_; ix < nx_; ++ix) {
        if (observed_iz[ix] >= 0) {
            manifold_iz_[ix]       = observed_iz[ix];
            manifold_z_[ix]        = grid_to_world_z(observed_iz[ix]);
            last_iz                = observed_iz[ix];
            manifold_observed_[ix] = true;
        } else if (last_iz >= 0) {
            manifold_iz_[ix]       = last_iz;
            manifold_z_[ix]        = grid_to_world_z(last_iz);
            manifold_observed_[ix] = true;
        } else {
            manifold_iz_[ix]       = bottom_iz;
            manifold_z_[ix]        = bottom_z;
            manifold_observed_[ix] = false;
        }
    }
}

// ===========================================================================
// Internal planning helpers
// ===========================================================================

OccupancyMap::ObstacleResult OccupancyMap::forward_obstacle(
    double vehicle_depth,
    double vehicle_world_x,
    double z_threshold) const
{
    const auto& c = cfg_;
    double nose_x = vehicle_world_x + c.vehicle_length / 2.0;
    double x_min  = vehicle_world_x;
    double x_max  = nose_x + c.cliff_standoff;
    double z_max  = std::isnan(z_threshold)
                    ? vehicle_depth + c.safety_below_m
                    : z_threshold;

    int ix_lo = std::max(cx_, static_cast<int>(std::floor(
                    (x_min - grid_origin_x_) / c.dx)));
    int ix_hi = std::min(nx_, static_cast<int>(std::ceil(
                    (x_max - grid_origin_x_) / c.dx)) + 1);

    double peak_z  = std::numeric_limits<double>::infinity();
    int    peak_ix = -1;

    for (int ix = ix_lo; ix < ix_hi; ++ix) {
        if (!manifold_observed_[ix]) continue;
        double col_x = grid_to_world_x(ix);
        if (col_x < x_min || col_x > x_max) continue;
        double z = manifold_z_[ix];
        if (z > z_max) continue;
        if (z < peak_z) {
            peak_z  = z;
            peak_ix = ix;
        }
    }

    if (peak_ix < 0) {
        return {false, kNaN, kNaN};
    }
    return {true, peak_z, grid_to_world_x(peak_ix)};
}

// Safety tail-check: find the first column behind the vehicle whose manifold
// segment crosses the depth band [v_z, v_z + safety_below_m) within
// (safety_standoff_m + vehicle_length/2) behind the centre.
bool OccupancyMap::safety_tail_blocked(double vehicle_depth) const {
    const auto& c = cfg_;
    double v_x    = grid_to_world_x(cx_);
    double tail_x = v_x - c.vehicle_length / 2.0;
    double x_min  = tail_x - c.safety_standoff_m;
    double x_max  = v_x;
    double z_lo   = vehicle_depth;
    double z_hi   = vehicle_depth + c.safety_below_m;

    int ix_lo = std::max(0, static_cast<int>(std::floor(
                    (x_min - grid_origin_x_) / c.dx)));
    int ix_hi = std::min(nx_, static_cast<int>(std::ceil(
                    (x_max - grid_origin_x_) / c.dx)) + 1);

    for (int ix = ix_lo; ix < ix_hi; ++ix) {
        if (!manifold_observed_[ix]) continue;
        double z = manifold_z_[ix];
        if (!(z >= z_lo && z < z_hi)) continue;
        double x = grid_to_world_x(ix);
        if (x >= x_min && x <= x_max) return true;
    }
    return false;
}

void OccupancyMap::set_mode_from_cmd_depth(double vehicle_depth) {
    const auto& c = cfg_;
    double target_z = cmd_depth_[cx_];

    if (std::isnan(target_z)) {
        control_mode_ = "ALT_FOLLOW";
        return;
    }

    double dz = target_z - vehicle_depth;
    double T     = static_cast<double>(c.altitude_overshoot_threshold_m);
    double h_raw = static_cast<double>(c.altitude_overshoot_hysteresis_m);
    double h     = std::max(0.0, std::min(h_raw, std::max(0.0, T - 1e-6)));
    double inside = std::max(0.0, T - h);
    const std::string& prev = control_mode_;

    bool obstacle_clear = (dz <= -T) || (prev == "OBSTACLE_CLEAR" && dz < -inside);
    bool altitude_correction = (dz >= T) || (prev == "ALT_CORRECTION" && dz > inside);

    if (obstacle_clear && altitude_correction) {
        if (dz <= -inside) {
            control_mode_ = "OBSTACLE_CLEAR";
        } else if (dz >= inside) {
            control_mode_ = "ALT_CORRECTION";
        } else {
            control_mode_ = "ALT_FOLLOW";
        }
        return;
    }
    if (obstacle_clear) {
        control_mode_ = "OBSTACLE_CLEAR";
        return;
    }
    if (altitude_correction) {
        control_mode_ = "ALT_CORRECTION";
        return;
    }
    control_mode_ = "ALT_FOLLOW";
}

// ===========================================================================
// build_commanded_depth
// ===========================================================================

void OccupancyMap::build_commanded_depth(double vehicle_depth) {
    const auto& c = cfg_;
    double vehicle_world_x =
        grid_origin_x_ + cx_ * c.dx + shift_accum_;

    // ----- Step 1: altitude-following baseline + DVL min override -----
    for (int ix = 0; ix < nx_; ++ix) {
        double z = manifold_z_[ix];
        if (std::isnan(z)) {
            cmd_depth_[ix] = kNaN;
        } else {
            cmd_depth_[ix] = std::max(0.0, z - c.imaging_altitude);
        }
    }
    // Vehicle column: the most conservative command the current knowledge
    // supports.  Two sources, and the shallower wins.
    //
    //   (a) the shallowest observed manifold across the whole hull footprint,
    //       not the single column under the origin.  On a slope the stern or
    //       bow sits over terrain the nadir column knows nothing about, and
    //       commanding to the nadir alone descends one end into the seabed.
    //
    //   (b) the latest direct vertical range — DVL min-across-beams, or the
    //       altimeter.  A current measurement is fresher than the manifold and
    //       survives a heading change that invalidates the map, so it must be
    //       able to override a manifold-derived command whenever it is the
    //       safer of the two.
    //
    // This deliberately reverses an earlier rule that let the DVL deepen the
    // command but never shallow it.  That rule protected survey throughput — a
    // forward Janus beam striking a cliff wall returns a short range, which
    // under this rule commands an ascent and stops forward motion.  Safety wins
    // that trade: an unnecessary climb costs survey time, descending onto
    // terrain does not fail gracefully.
    int half_bins = static_cast<int>(std::ceil(c.vehicle_length / (2.0 * c.dx)));
    int foot_lo   = std::max(0, cx_ - half_bins);
    int foot_hi   = std::min(nx_ - 1, cx_ + half_bins);
    double foot_z = kNaN;
    for (int i = foot_lo; i <= foot_hi; ++i) {
        if (!manifold_observed_[i]) continue;          // unobserved => grid floor
        double z = manifold_z_[i];
        if (std::isnan(z)) continue;
        if (std::isnan(foot_z) || z < foot_z) foot_z = z;
    }

    double measured_alt = dvl_altitude_;
    if (!std::isnan(altimeter_altitude_)) {
        measured_alt = std::isnan(measured_alt)
                     ? altimeter_altitude_
                     : std::min(measured_alt, altimeter_altitude_);
    }

    double vcmd = kNaN;
    if (!std::isnan(foot_z)) {
        vcmd = std::max(0.0, foot_z - c.imaging_altitude);
    }
    if (!std::isnan(measured_alt)) {
        double mcmd = std::max(0.0, vehicle_depth + measured_alt - c.imaging_altitude);
        vcmd = std::isnan(vcmd) ? mcmd : std::min(vcmd, mcmd);
    }
    if (std::isnan(vcmd)) {
        // Neither a direct return nor a real manifold observation anywhere under
        // the hull (the grid defaults to bottom depth).  Do not command a dive
        // to the depth-window floor — that would force ALT_CORRECTION with zero
        // forward motion.  Hold at the current depth until something sees
        // seafloor.
        vcmd = std::max(0.0, vehicle_depth);
    }
    cmd_depth_[cx_] = vcmd;

    // ----- Step 2a: release latch if past tracked obstacle + margin -----
    if (cliff_top_committed_ && vehicle_world_x >= cliff_top_release_x_) {
        cliff_top_committed_ = false;
        cliff_top_target_z_  = kNaN;
        cliff_top_target_x_  = kNaN;
        cliff_top_release_x_ = kNaN;
        cliff_top_commit_heading_ = kNaN;
    }

    // ----- Step 2b: forward-obstacle detection -----
    auto obs = forward_obstacle(vehicle_depth, vehicle_world_x);
    if (obs.found) {
        // Deliberately not clamped to 0.  If peak_z - imaging_altitude is
        // negative the obstacle top is above the water surface, and the
        // vehicle will try to ascend past it while the platform's z >= 0 limit
        // holds it there.  That is the intended terminal behaviour on a slope
        // too steep to traverse: the controller fails safe at the surface
        // rather than commanding a depth that would take the vehicle through
        // terrain.  It is the one state the planner does not exit on its own —
        // escaping it is an operator/mission-level decision.
        double new_target_z = obs.peak_z - c.imaging_altitude;
        if (!cliff_top_committed_) {
            cliff_top_committed_ = true;
            cliff_top_target_z_  = new_target_z;
            cliff_top_target_x_  = obs.peak_world_x;
            cliff_top_commit_heading_ = last_vehicle_heading_;
        } else {
            if (new_target_z < cliff_top_target_z_) cliff_top_target_z_ = new_target_z;
            if (obs.peak_world_x > cliff_top_target_x_) {
                cliff_top_target_x_ = obs.peak_world_x;
                // Re-anchor the commit heading: the latch now tracks a peak
                // observed at the current heading, so the turn-away release
                // must be measured from here, not from the original commit.
                cliff_top_commit_heading_ = last_vehicle_heading_;
            }
        }
        cliff_top_release_x_ = cliff_top_target_x_ + c.cliff_standoff + c.vehicle_length;
    } else if (cliff_top_committed_) {
        // Wide scan using imaging_altitude as depth threshold
        auto wide = forward_obstacle(vehicle_depth, vehicle_world_x,
                                     vehicle_depth + c.imaging_altitude);
        if (wide.found) {
            double new_target_z = wide.peak_z - c.imaging_altitude;
            if (new_target_z < cliff_top_target_z_) {
                cliff_top_target_z_ = new_target_z;
                if (wide.peak_world_x > cliff_top_target_x_) {
                    cliff_top_target_x_ = wide.peak_world_x;
                    cliff_top_commit_heading_ = last_vehicle_heading_;
                }
                cliff_top_release_x_ = cliff_top_target_x_ + c.cliff_standoff + c.vehicle_length;
            }
            // else: terrain at or below imaging_altitude — keep latch so the
            // tail clears the highest obstacle voxel before the altimeter
            // takes over.
        } else {
            // Both narrow and wide scans empty: no forward obstacle visible.
            // Only release if the vehicle has turned significantly from the
            // heading the latch anchored at (same threshold as
            // clear_stale_voxels), which is what causes the occupied voxels to
            // disappear.  Without this guard the cliff peak passing behind
            // vehicle_center during normal OBSTACLE_HOLD forward flight would
            // trigger a premature release before the tail has cleared.
            double h_cur = last_vehicle_heading_;
            double h_cmt = cliff_top_commit_heading_;
            if (!std::isnan(h_cur) && !std::isnan(h_cmt)) {
                // Same wrap handling as clear_stale_voxels: the fmod first
                // keeps this correct for unwrapped headings, where a bare
                // (2*pi - diff) would go negative.
                double diff = std::fmod(std::abs(h_cur - h_cmt), 2.0 * M_PI);
                diff = std::min(diff, 2.0 * M_PI - diff);
                if (diff > c.stale_heading_threshold_deg * M_PI / 180.0) {
                    cliff_top_committed_ = false;
                    cliff_top_target_z_  = kNaN;
                    cliff_top_target_x_  = kNaN;
                    cliff_top_release_x_ = kNaN;
                    cliff_top_commit_heading_ = kNaN;
                }
            }
        }
    }

    // ----- Step 2c: apply latch -----
    if (cliff_top_committed_) {
        double effective = cliff_top_target_z_;
        for (int ix = cx_; ix < nx_; ++ix) {
            cmd_depth_[ix] = effective;
        }
        set_mode_from_cmd_depth(vehicle_depth);
        if (control_mode_ != "OBSTACLE_CLEAR") {
            control_mode_ = "OBSTACLE_HOLD";
        }
        return;
    }

    // ----- Step 3: safety tail cap -----
    if (safety_tail_blocked(vehicle_depth)) {
        for (int ix = cx_; ix < nx_; ++ix) {
            if (cmd_depth_[ix] > vehicle_depth) {
                cmd_depth_[ix] = vehicle_depth;
            }
        }
    }

    // ----- Step 4: propagate vehicle-column depth to unobserved ahead columns -----
    for (int ix = cx_ + 1; ix < nx_; ++ix) {
        if (!manifold_observed_[ix]) {
            cmd_depth_[ix] = cmd_depth_[cx_];
        }
    }

    // ----- Mode selection -----
    // Tail clearance overrides altitude follow / correction only; the forward
    // obstacle latch (which returns earlier) and OBSTACLE_CLEAR are unchanged.
    set_mode_from_cmd_depth(vehicle_depth);
    if (safety_tail_blocked(vehicle_depth) &&
        (control_mode_ == "ALT_FOLLOW" || control_mode_ == "ALT_CORRECTION")) {
        control_mode_ = "TAIL_CLEAR";
    }
}

// ===========================================================================
// build_path_waypoints
// ===========================================================================

void OccupancyMap::build_path_waypoints() {
    const auto& c = cfg_;
    double step_thresh = 2.0 * c.dz;
    path_waypoints_.clear();

    for (int ix = cx_; ix < nx_; ++ix) {
        double world_x = grid_to_world_x(ix);
        double depth   = cmd_depth_[ix];

        if (std::isnan(depth)) continue;

        if (path_waypoints_.empty()) {
            path_waypoints_.push_back({world_x, depth});
            continue;
        }

        auto [prev_x, prev_z] = path_waypoints_.back();
        double dz = std::abs(depth - prev_z);

        if (dz > step_thresh) {
            if (std::abs(world_x - prev_x) > 0.01) {
                path_waypoints_.push_back({world_x, prev_z});
            }
            path_waypoints_.push_back({world_x, depth});
        } else {
            path_waypoints_.push_back({world_x, depth});
        }
    }
}

double OccupancyMap::get_commanded_depth_at_vehicle() const {
    if (cx_ >= 0 && cx_ < nx_) return cmd_depth_[cx_];
    return kNaN;
}

// ===========================================================================
// clear_stale_voxels
// ===========================================================================

void OccupancyMap::clear_stale_voxels(double vehicle_heading) {
    if (std::isnan(vehicle_heading)) return;

    const auto& c = cfg_;

    // Refresh heading for occupied voxels under the vehicle body
    int half_bins = static_cast<int>(std::ceil(c.vehicle_length / (2.0 * c.dx)));
    int ix_lo = std::max(0, cx_ - half_bins);
    int ix_hi = std::min(nx_, cx_ + half_bins + 1);

    for (int ix = ix_lo; ix < ix_hi; ++ix) {
        for (int iz = 0; iz < nz_; ++iz) {
            if (grid_(iz, ix) >= c.occ_thresh) {
                voxel_heading_(iz, ix) = vehicle_heading;
            }
        }
    }

    double threshold = c.stale_heading_threshold_deg * M_PI / 180.0;

    for (int ix = 0; ix < nx_; ++ix) {
        for (int iz = 0; iz < nz_; ++iz) {
            double h = voxel_heading_(iz, ix);
            if (std::isnan(h)) continue;
            double diff = std::fmod(std::abs(h - vehicle_heading), 2.0 * M_PI);
            if (diff > M_PI) diff = 2.0 * M_PI - diff;
            if (diff > threshold) {
                grid_(iz, ix)         = c.prior;
                voxel_heading_(iz, ix) = kNaN;
            }
        }
    }
}

// ===========================================================================
// update — full pipeline
// ===========================================================================

double OccupancyMap::update(double vehicle_depth, double vehicle_heading) {
    last_vehicle_heading_ = vehicle_heading;
    clear_stale_voxels(vehicle_heading);
    build_cliff_manifold();
    build_commanded_depth(vehicle_depth);
    build_path_waypoints();
    return get_commanded_depth_at_vehicle();
}

// ===========================================================================
// get_debug_summary
// ===========================================================================

std::string OccupancyMap::get_debug_summary(double vehicle_depth) const {
    const auto& c = cfg_;
    int cx = cx_;
    int win = 6;

    auto fmt = [&](const Eigen::VectorXd& arr, int col) -> std::string {
        if (col >= 0 && col < nx_) {
            double v = arr[col];
            if (std::isnan(v)) return "   nan";
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << std::showpos << std::setw(6) << v;
            return oss.str();
        }
        return "  ---";
    };

    // Collect column indices for display window
    std::vector<int> idxs;
    for (int i = std::max(0, cx - win); i < std::min(nx_, cx + win + 1); ++i) {
        idxs.push_back(i);
    }

    // Build header row
    std::ostringstream hdr_ss;
    for (int k = 0; k < static_cast<int>(idxs.size()); ++k) {
        if (k) hdr_ss << "  ";
        int offset = idxs[k] - cx;
        std::string label = (offset == 0) ? "cx" : ("cx" + std::to_string(offset));
        hdr_ss << std::setw(6) << std::right << label;
    }

    // World X row
    std::ostringstream xrow_ss;
    for (int k = 0; k < static_cast<int>(idxs.size()); ++k) {
        if (k) xrow_ss << "  ";
        double wx = grid_to_world_x(idxs[k]);
        std::ostringstream tmp;
        tmp << std::fixed << std::setprecision(1) << std::showpos << std::setw(6) << wx;
        xrow_ss << tmp.str();
    }

    // Manifold row
    std::ostringstream mrow_ss;
    for (int k = 0; k < static_cast<int>(idxs.size()); ++k) {
        if (k) mrow_ss << "  ";
        mrow_ss << fmt(manifold_z_, idxs[k]);
    }

    // Cmd depth row
    std::ostringstream crow_ss;
    for (int k = 0; k < static_cast<int>(idxs.size()); ++k) {
        if (k) crow_ss << "  ";
        crow_ss << fmt(cmd_depth_, idxs[k]);
    }

    // Forward-obstacle detection
    double vehicle_world_x = grid_origin_x_ + cx_ * c.dx + shift_accum_;
    auto obs = forward_obstacle(vehicle_depth, vehicle_world_x);
    std::string climb_str;
    if (obs.found) {
        double climb_target = std::max(0.0, obs.peak_z - c.imaging_altitude);
        std::ostringstream tmp;
        tmp << "target=" << std::fixed << std::setprecision(2) << climb_target << "m";
        climb_str = tmp.str();
    } else {
        climb_str = "no rise in window";
    }

    // Safety tail check
    int safety_bins = static_cast<int>(std::ceil(
        (c.safety_standoff_m + c.vehicle_length / 2.0) / c.dx));
    int safety_ix = -1;
    double z_lo = vehicle_depth;
    double z_hi = vehicle_depth + c.safety_below_m;
    for (int ix = std::max(0, cx - safety_bins); ix < cx; ++ix) {
        if (ix + 1 >= nx_) continue;
        if (!manifold_observed_[ix] || !manifold_observed_[ix + 1]) continue;
        double s_lo = std::min(manifold_z_[ix], manifold_z_[ix + 1]);
        double s_hi = std::max(manifold_z_[ix], manifold_z_[ix + 1]);
        if (s_hi >= z_lo && s_lo < z_hi) {
            safety_ix = ix;
            break;
        }
    }

    // Waypoints (first 6)
    std::ostringstream wp_ss;
    int n_wp = std::min(static_cast<int>(path_waypoints_.size()), 6);
    for (int k = 0; k < n_wp; ++k) {
        if (k) wp_ss << "  ";
        auto [wx, wz] = path_waypoints_[k];
        wp_ss << "(" << std::fixed << std::setprecision(1) << std::showpos << wx
              << "," << std::noshowpos << std::fixed << std::setprecision(1) << wz << ")";
    }

    // DVL altitude string
    std::string dvl_str;
    if (!std::isnan(dvl_altitude_)) {
        std::ostringstream tmp;
        tmp << std::fixed << std::setprecision(3) << dvl_altitude_ << "m";
        dvl_str = tmp.str();
    } else {
        dvl_str = "nan";
    }

    double cmd_at_cx = (cx_ >= 0 && cx_ < nx_) ? cmd_depth_[cx_] : kNaN;

    std::ostringstream out;
    out << "  mode=" << std::left << std::setw(15) << control_mode_
        << "  dvl_alt=" << dvl_str
        << "  target=" << std::fixed << std::setprecision(2) << c.imaging_altitude << "m"
        << "  veh_z=" << std::fixed << std::setprecision(3) << vehicle_depth << "m"
        << "  cmd@cx=" << std::fixed << std::setprecision(3) << cmd_at_cx << "m\n"
        << "  col offset: " << hdr_ss.str() << "\n"
        << "  world_x:    " << xrow_ss.str() << "\n"
        << "  manifold_z: " << mrow_ss.str() << "\n"
        << "  cmd_depth:  " << crow_ss.str() << "\n"
        << "  climb:  " << climb_str << "\n"
        << "  safety: ";
    if (safety_ix >= 0) {
        out << "manifold in band at col cx" << (safety_ix - cx)
            << " (x=" << std::fixed << std::setprecision(2) << std::showpos
            << grid_to_world_x(safety_ix) << "m)";
    } else {
        out << "clear";
    }
    out << "\n  waypoints[0:6]: " << (wp_ss.str().empty() ? "(none)" : wp_ss.str());

    return out.str();
}

// ===========================================================================
// ObstacleMapper
// ===========================================================================

ObstacleMapper::ObstacleMapper(
    const OccupancyMapConfig& cfg,
    const DVLConfig&          dvl_config,
    const SonarConfig&        sonar_config,
    const AltimeterConfig&    altimeter_config)
    : omap_(cfg)
    , dvl_config_(dvl_config)
    , sonar_config_(sonar_config)
    , altimeter_config_(altimeter_config)
    , last_pose_(std::nullopt)
    , altimeter_altitude_(kNaN)
{
}

void ObstacleMapper::reset(const Pose& pose) {
    std::lock_guard<std::mutex> guard(lock_);
    omap_.reset(0.0, pose.depth);
    last_pose_            = pose;
    altimeter_altitude_   = kNaN;
}

void ObstacleMapper::advance_to_pose(const Pose& pose) {
    if (!last_pose_.has_value()) {
        omap_.reset(0.0, pose.depth);
        last_pose_ = pose;
        return;
    }
    double dn = pose.north - last_pose_->north;
    double de = pose.east  - last_pose_->east;

    // Detect mission restart / teleport: if the straight-line jump exceeds the
    // grid length the vehicle is outside the mapped area.  Reset rather than
    // projecting a huge (and mis-signed) ds onto the arc-local axis.
    double grid_len = (omap_.nx() - 1) * omap_.cfg().dx;
    if (dn*dn + de*de > grid_len * grid_len) {
        omap_.reset(0.0, pose.depth);
        last_pose_ = pose;
        return;
    }

    double h  = last_pose_->heading;
    double ds = dn * std::cos(h) + de * std::sin(h);
    if (ds > 0.0) omap_.advance(ds);
    omap_.shift_depth(pose.depth);
    last_pose_ = pose;
}

double ObstacleMapper::vehicle_forward_x() const {
    return omap_.grid_origin_x()
         + omap_.cx() * omap_.cfg().dx
         + omap_.shift_accum();
}

void ObstacleMapper::update_sensor(SensorType /*type*/, const DVLMeasurement& meas, const Pose& pose) {
    std::lock_guard<std::mutex> guard(lock_);
    advance_to_pose(pose);
    double fwd_x  = vehicle_forward_x();
    auto angles    = dvl_config_.beam_angles_rad();
    auto can_clear = dvl_config_.beam_can_clear();
    omap_.update_dvl_ray(meas.ranges, angles, pose.depth, fwd_x,
                         std::optional<std::vector<bool>>(meas.hit_surface),
                         0.15, pose.heading,
                         std::optional<std::vector<bool>>(can_clear));
    omap_.update(pose.depth, pose.heading);
}

void ObstacleMapper::update_sensor(SensorType /*type*/, const AltimeterMeasurement& meas, const Pose& pose) {
    std::lock_guard<std::mutex> guard(lock_);
    advance_to_pose(pose);
    double fwd_x = vehicle_forward_x();

    bool valid = meas.hit && (meas.range_m < altimeter_config_.max_range - 0.05);
    altimeter_altitude_ = valid ? meas.range_m : kNaN;
    omap_.set_altimeter_altitude(altimeter_altitude_);

    omap_.update_altimeter_ray(meas.range_m, pose.depth, fwd_x,
                               meas.hit, 0.15, pose.heading);
    omap_.update(pose.depth, pose.heading);
}

void ObstacleMapper::update_sensor(SensorType /*type*/, const SonarMeasurement& meas, const Pose& pose) {
    std::lock_guard<std::mutex> guard(lock_);
    advance_to_pose(pose);
    double fwd_x  = vehicle_forward_x();
    double nose_x = fwd_x + omap_.cfg().vehicle_length / 2.0;
    omap_.update_sonar(meas.range_m, sonar_config_.half_angle_rad(),
                       pose.depth, nose_x, meas.hit,
                       0.2, 7, pose.heading);
    omap_.update(pose.depth, pose.heading);
}

void ObstacleMapper::update_pose(const Pose& pose) {
    std::lock_guard<std::mutex> guard(lock_);
    advance_to_pose(pose);
    omap_.update(pose.depth, pose.heading);
}

ControlCommand ObstacleMapper::get_control() {
    std::lock_guard<std::mutex> guard(lock_);
    if (!last_pose_.has_value()) {
        return {omap_.cfg().survey_speed, "ALT_FOLLOW", omap_.cfg().imaging_altitude};
    }

    const auto& mode = omap_.control_mode();
    const auto& c    = omap_.cfg();

    if (mode == "OBSTACLE_CLEAR") {
        double cmd_depth  = omap_.get_commanded_depth_at_vehicle();
        double target_depth = (!std::isnan(cmd_depth)) ? cmd_depth : last_pose_->depth;
        double dz  = target_depth - last_pose_->depth;
        double vx  = (std::abs(dz) > 0.1) ? 0.0 : c.survey_speed;
        return {vx, "DEPTH_HOLD", target_depth};
    }
    if (mode == "OBSTACLE_HOLD") {
        double cmd_depth  = omap_.get_commanded_depth_at_vehicle();
        double target_depth = (!std::isnan(cmd_depth)) ? cmd_depth : last_pose_->depth;
        return {c.survey_speed, "DEPTH_HOLD", target_depth};
    }
    if (mode == "ALT_CORRECTION") {
        double cmd_depth = omap_.get_commanded_depth_at_vehicle();
        double target = (!std::isnan(cmd_depth)) ? cmd_depth : last_pose_->depth;
        return {0.0, "DEPTH_HOLD", target};
    }
    if (mode == "TAIL_CLEAR") {
        return {c.survey_speed, "DEPTH_HOLD", last_pose_->depth};
    }
    // ALT_FOLLOW
    return {c.survey_speed, "ALT_FOLLOW", c.imaging_altitude};
}

double ObstacleMapper::get_altitude() {
    std::lock_guard<std::mutex> guard(lock_);
    double dvl_alt = omap_.dvl_altitude();
    // return minimum of valid readings
    bool dvl_valid = !std::isnan(dvl_alt);
    bool alt_valid = !std::isnan(altimeter_altitude_);
    if (!dvl_valid && !alt_valid) return kNaN;
    if (!dvl_valid) return altimeter_altitude_;
    if (!alt_valid) return dvl_alt;
    return std::min(dvl_alt, altimeter_altitude_);
}

} // namespace auv_oavoid
