#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>

#include "occupancy_map.hpp"

namespace py = pybind11;
using namespace auv_oavoid;

// ---------------------------------------------------------------------------
// Helper: convert std::vector<bool> to numpy bool array
// ---------------------------------------------------------------------------
static py::array_t<bool> vec_bool_to_numpy(const std::vector<bool>& v) {
    py::array_t<bool> arr(static_cast<py::ssize_t>(v.size()));
    auto buf = arr.mutable_unchecked<1>();
    for (size_t i = 0; i < v.size(); ++i) buf[i] = v[i];
    return arr;
}

// ---------------------------------------------------------------------------
// Helper: convert py::object (None | numpy bool array) to optional<vector<bool>>
// ---------------------------------------------------------------------------
static std::optional<std::vector<bool>> obj_to_opt_bool(py::object obj) {
    if (obj.is_none()) return std::nullopt;
    // Use ensure() to get a contiguous bool array regardless of input type.
    auto arr = py::array_t<bool>::ensure(obj);
    if (!arr) {
        throw py::type_error("hit_surface must be a numpy bool array or None");
    }
    auto r = arr.unchecked<1>();
    std::vector<bool> result(static_cast<size_t>(r.shape(0)));
    for (py::ssize_t i = 0; i < r.shape(0); ++i) result[static_cast<size_t>(i)] = r[i];
    return result;
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------
PYBIND11_MODULE(occupancy_map_cpp, m) {
    m.doc() = "C++ port of AUV obstacle avoidance occupancy map (auv_oavoid)";

    // -----------------------------------------------------------------------
    // OccupancyMapConfig
    // -----------------------------------------------------------------------
    py::class_<OccupancyMapConfig>(m, "OccupancyMapConfig")
        .def(py::init<>())
        // Grid dimensions
        .def_readwrite("dx",            &OccupancyMapConfig::dx)
        .def_readwrite("dz",            &OccupancyMapConfig::dz)
        .def_readwrite("horizon_fwd",   &OccupancyMapConfig::horizon_fwd)
        .def_readwrite("horizon_back",  &OccupancyMapConfig::horizon_back)
        .def_readwrite("z_half_range",  &OccupancyMapConfig::z_half_range)
        // Vehicle
        .def_readwrite("vehicle_length",    &OccupancyMapConfig::vehicle_length)
        .def_readwrite("imaging_altitude",  &OccupancyMapConfig::imaging_altitude)
        .def_readwrite("survey_speed",      &OccupancyMapConfig::survey_speed)
        .def_readwrite("vertical_speed",    &OccupancyMapConfig::vertical_speed)
        // Path planning
        .def_readwrite("cliff_standoff",        &OccupancyMapConfig::cliff_standoff)
        .def_readwrite("obstacle_threshold",    &OccupancyMapConfig::obstacle_threshold)
        .def_readwrite("altitude_overshoot_threshold_m",  &OccupancyMapConfig::altitude_overshoot_threshold_m)
        .def_readwrite("altitude_overshoot_hysteresis_m", &OccupancyMapConfig::altitude_overshoot_hysteresis_m)
        .def_readwrite("stale_heading_threshold_deg", &OccupancyMapConfig::stale_heading_threshold_deg)
        // Safety clearance
        .def_readwrite("safety_standoff_m",  &OccupancyMapConfig::safety_standoff_m)
        .def_readwrite("safety_below_m",     &OccupancyMapConfig::safety_below_m)
        // Occupancy probability
        .def_readwrite("prior",      &OccupancyMapConfig::prior)
        .def_readwrite("occ_thresh", &OccupancyMapConfig::occ_thresh)
        // DVL model
        .def_readwrite("dvl_hit_prob",    &OccupancyMapConfig::dvl_hit_prob)
        .def_readwrite("dvl_miss_prob",   &OccupancyMapConfig::dvl_miss_prob)
        .def_readwrite("dvl_max_occ",     &OccupancyMapConfig::dvl_max_occ)
        .def_readwrite("dvl_min_occ",     &OccupancyMapConfig::dvl_min_occ)
        .def_readwrite("dvl_max_range_m", &OccupancyMapConfig::dvl_max_range_m)
        // Altimeter model
        .def_readwrite("altimeter_hit_prob",  &OccupancyMapConfig::altimeter_hit_prob)
        .def_readwrite("altimeter_miss_prob", &OccupancyMapConfig::altimeter_miss_prob)
        .def_readwrite("altimeter_max_occ",   &OccupancyMapConfig::altimeter_max_occ)
        .def_readwrite("altimeter_min_occ",   &OccupancyMapConfig::altimeter_min_occ)
        // Sonar model
        .def_readwrite("sonar_hit_prob",    &OccupancyMapConfig::sonar_hit_prob)
        .def_readwrite("sonar_miss_prob",   &OccupancyMapConfig::sonar_miss_prob)
        .def_readwrite("sonar_max_occ",     &OccupancyMapConfig::sonar_max_occ)
        .def_readwrite("sonar_min_occ",     &OccupancyMapConfig::sonar_min_occ)
        .def_readwrite("sonar_min_depth_m", &OccupancyMapConfig::sonar_min_depth_m)
    ;

    // -----------------------------------------------------------------------
    // DVLConfig
    // -----------------------------------------------------------------------
    py::class_<DVLConfig>(m, "DVLConfig")
        .def(py::init<>())
        .def_readwrite("beams",     &DVLConfig::beams)
        .def_readwrite("max_range", &DVLConfig::max_range)
        .def_property_readonly("beam_angles_rad",
            [](const DVLConfig& self) {
                auto v = self.beam_angles_rad();
                py::array_t<double> arr(static_cast<py::ssize_t>(v.size()));
                auto buf = arr.mutable_unchecked<1>();
                for (py::ssize_t i = 0; i < static_cast<py::ssize_t>(v.size()); ++i)
                    buf[i] = v[static_cast<size_t>(i)];
                return arr;
            })
        .def_property_readonly("beam_directions_3d",
            [](const DVLConfig& self) -> Eigen::MatrixXd {
                return self.beam_directions_3d();
            })
        .def_property_readonly("beam_can_clear",
            [](const DVLConfig& self) {
                return vec_bool_to_numpy(self.beam_can_clear());
            })
    ;

    // -----------------------------------------------------------------------
    // SonarConfig
    // -----------------------------------------------------------------------
    py::class_<SonarConfig>(m, "SonarConfig")
        .def(py::init<>())
        .def_readwrite("max_range",  &SonarConfig::max_range)
        .def_readwrite("half_angle", &SonarConfig::half_angle)
        .def_readwrite("noise_std",  &SonarConfig::noise_std)
        .def_property_readonly("half_angle_rad",
            [](const SonarConfig& self) { return self.half_angle_rad(); })
    ;

    // -----------------------------------------------------------------------
    // AltimeterConfig
    // -----------------------------------------------------------------------
    py::class_<AltimeterConfig>(m, "AltimeterConfig")
        .def(py::init<>())
        .def_readwrite("max_range", &AltimeterConfig::max_range)
    ;

    // -----------------------------------------------------------------------
    // Pose
    // -----------------------------------------------------------------------
    py::class_<Pose>(m, "Pose")
        .def(py::init<double, double, double, double>(),
             py::arg("north"), py::arg("east"), py::arg("depth"), py::arg("heading"))
        .def_readwrite("north",   &Pose::north)
        .def_readwrite("east",    &Pose::east)
        .def_readwrite("depth",   &Pose::depth)
        .def_readwrite("heading", &Pose::heading)
    ;

    // -----------------------------------------------------------------------
    // ControlCommand
    // -----------------------------------------------------------------------
    py::class_<ControlCommand>(m, "ControlCommand")
        .def(py::init<double, std::string, double>(),
             py::arg("vx"), py::arg("vertical_mode"), py::arg("vertical_target"))
        .def_readwrite("vx",              &ControlCommand::vx)
        .def_readwrite("vertical_mode",   &ControlCommand::vertical_mode)
        .def_readwrite("vertical_target", &ControlCommand::vertical_target)
        .def("__repr__", [](const ControlCommand& cmd) {
            return "ControlCommand(vx=" + std::to_string(cmd.vx)
                 + ", vertical_mode='" + cmd.vertical_mode
                 + "', vertical_target=" + std::to_string(cmd.vertical_target) + ")";
        })
    ;

    // -----------------------------------------------------------------------
    // SensorType enum
    // -----------------------------------------------------------------------
    py::enum_<SensorType>(m, "SensorType")
        .value("DVL",       SensorType::DVL)
        .value("ALTIMETER", SensorType::ALTIMETER)
        .value("SONAR",     SensorType::SONAR)
        .export_values()
    ;

    // -----------------------------------------------------------------------
    // DVLMeasurement
    // -----------------------------------------------------------------------
    py::class_<DVLMeasurement>(m, "DVLMeasurement")
        .def(py::init([](py::array_t<double> ranges_arr, py::array_t<bool> hit_arr) {
            auto r = ranges_arr.unchecked<1>();
            auto h = hit_arr.unchecked<1>();
            if (r.shape(0) != h.shape(0))
                throw std::invalid_argument("DVLMeasurement: ranges and hit_surface must be the same length");
            DVLMeasurement m;
            m.ranges.resize(static_cast<size_t>(r.shape(0)));
            m.hit_surface.resize(static_cast<size_t>(h.shape(0)));
            for (py::ssize_t i = 0; i < r.shape(0); ++i)
                m.ranges[static_cast<size_t>(i)] = r[i];
            for (py::ssize_t i = 0; i < h.shape(0); ++i)
                m.hit_surface[static_cast<size_t>(i)] = h[i];
            return m;
        }), py::arg("ranges"), py::arg("hit_surface"))
        .def_readwrite("ranges",      &DVLMeasurement::ranges)
        .def_readwrite("hit_surface", &DVLMeasurement::hit_surface)
    ;

    // -----------------------------------------------------------------------
    // AltimeterMeasurement
    // -----------------------------------------------------------------------
    py::class_<AltimeterMeasurement>(m, "AltimeterMeasurement")
        .def(py::init<double, bool>(), py::arg("range_m"), py::arg("hit") = true)
        .def_readwrite("range_m", &AltimeterMeasurement::range_m)
        .def_readwrite("hit",     &AltimeterMeasurement::hit)
    ;

    // -----------------------------------------------------------------------
    // SonarMeasurement
    // -----------------------------------------------------------------------
    py::class_<SonarMeasurement>(m, "SonarMeasurement")
        .def(py::init<double, bool>(), py::arg("range_m"), py::arg("hit"))
        .def_readwrite("range_m", &SonarMeasurement::range_m)
        .def_readwrite("hit",     &SonarMeasurement::hit)
    ;

    // -----------------------------------------------------------------------
    // OccupancyMap
    // -----------------------------------------------------------------------
    py::class_<OccupancyMap>(m, "OccupancyMap")
        .def(py::init<const OccupancyMapConfig&>(),
             py::arg("cfg") = OccupancyMapConfig{})
        .def("reset", &OccupancyMap::reset,
             py::arg("vehicle_world_x") = 0.0, py::arg("vehicle_depth") = 0.0)

        // Coordinate transforms
        .def("world_to_grid",  &OccupancyMap::world_to_grid,
             py::arg("wx"), py::arg("wz"))
        .def("grid_to_world_x", &OccupancyMap::grid_to_world_x, py::arg("ix"))
        .def("grid_to_world_z", &OccupancyMap::grid_to_world_z, py::arg("iz"))

        // Grid management
        .def("advance",     &OccupancyMap::advance,     py::arg("dx"))
        .def("shift_depth", &OccupancyMap::shift_depth, py::arg("vehicle_z"))

        // Sensor updates — expose update_dvl_ray with py::object for hit_surface
        .def("update_dvl_ray",
            [](OccupancyMap& self,
               py::array_t<double> ranges_arr,
               py::array_t<double> beam_angles_arr,
               double vehicle_depth,
               double vehicle_world_x,
               py::object hit_surface_obj,
               double range_step,
               double vehicle_heading,
               py::object can_clear_obj)
            {
                auto r = ranges_arr.unchecked<1>();
                auto a = beam_angles_arr.unchecked<1>();
                std::vector<double> ranges(static_cast<size_t>(r.shape(0)));
                std::vector<double> angles(static_cast<size_t>(a.shape(0)));
                for (py::ssize_t i = 0; i < r.shape(0); ++i)
                    ranges[static_cast<size_t>(i)] = r[i];
                for (py::ssize_t i = 0; i < a.shape(0); ++i)
                    angles[static_cast<size_t>(i)] = a[i];

                auto hit_opt   = obj_to_opt_bool(hit_surface_obj);
                auto clear_opt = obj_to_opt_bool(can_clear_obj);
                self.update_dvl_ray(ranges, angles, vehicle_depth, vehicle_world_x,
                                    hit_opt, range_step, vehicle_heading, clear_opt);
            },
            py::arg("ranges"),
            py::arg("beam_angles"),
            py::arg("vehicle_depth"),
            py::arg("vehicle_world_x"),
            py::arg("hit_surface") = py::none(),
            py::arg("range_step") = 0.15,
            py::arg("vehicle_heading") = kNaN,
            py::arg("can_clear") = py::none())

        .def("update_altimeter_ray", &OccupancyMap::update_altimeter_ray,
             py::arg("range_m"), py::arg("vehicle_depth"), py::arg("vehicle_world_x"),
             py::arg("hit"), py::arg("range_step") = 0.15,
             py::arg("vehicle_heading") = kNaN)

        .def("update_sonar", &OccupancyMap::update_sonar,
             py::arg("sonar_range"), py::arg("sonar_half_angle"),
             py::arg("vehicle_depth"), py::arg("vehicle_world_x"),
             py::arg("hit_obstacle"),
             py::arg("range_step") = 0.2, py::arg("angle_steps") = 7,
             py::arg("vehicle_heading") = kNaN)

        // Planning pipeline
        .def("build_cliff_manifold",         &OccupancyMap::build_cliff_manifold)
        .def("build_commanded_depth",        &OccupancyMap::build_commanded_depth,
             py::arg("vehicle_depth"))
        .def("build_path_waypoints",         &OccupancyMap::build_path_waypoints)
        .def("get_commanded_depth_at_vehicle", &OccupancyMap::get_commanded_depth_at_vehicle)
        .def("clear_stale_voxels",           &OccupancyMap::clear_stale_voxels,
             py::arg("vehicle_heading"))
        .def("update",                       &OccupancyMap::update,
             py::arg("vehicle_depth"), py::arg("vehicle_heading") = kNaN)

        // Accessors — all exposed for Python simulator compatibility
        .def_property("cfg",
            [](OccupancyMap& self) -> OccupancyMapConfig& { return self.cfg(); },
            [](OccupancyMap& self, const OccupancyMapConfig& c) { self.cfg() = c; },
            py::return_value_policy::reference_internal)
        .def_property_readonly("nx", &OccupancyMap::nx)
        .def_property_readonly("nz", &OccupancyMap::nz)
        .def_property_readonly("cx", &OccupancyMap::cx)
        .def_property_readonly("grid_origin_x",          &OccupancyMap::grid_origin_x)
        .def_property_readonly("grid_origin_z",          &OccupancyMap::grid_origin_z)
        .def_property_readonly("manifold_grid_origin_x", &OccupancyMap::manifold_grid_origin_x)
        .def_property_readonly("dvl_altitude",           &OccupancyMap::dvl_altitude)
        .def_property_readonly("control_mode",           &OccupancyMap::control_mode)
        .def_property_readonly("shift_accum",            &OccupancyMap::shift_accum)

        // Grid and manifold arrays — return references (no copy) where possible
        .def_property_readonly("grid",
            [](const OccupancyMap& self) -> const Eigen::MatrixXd& {
                return self.grid();
            }, py::return_value_policy::reference_internal)

        .def_property_readonly("manifold_z",
            [](const OccupancyMap& self) -> const Eigen::VectorXd& {
                return self.manifold_z();
            }, py::return_value_policy::reference_internal)

        .def_property_readonly("manifold_iz",
            [](const OccupancyMap& self) -> const Eigen::VectorXi& {
                return self.manifold_iz();
            }, py::return_value_policy::reference_internal)

        .def_property_readonly("manifold_observed",
            [](const OccupancyMap& self) {
                return vec_bool_to_numpy(self.manifold_observed());
            })

        .def_property_readonly("cmd_depth",
            [](const OccupancyMap& self) -> const Eigen::VectorXd& {
                return self.cmd_depth();
            }, py::return_value_policy::reference_internal)

        .def_property_readonly("path_waypoints",
            [](const OccupancyMap& self) -> py::list {
                py::list out;
                for (auto& [x, z] : self.path_waypoints()) {
                    out.append(py::make_tuple(x, z));
                }
                return out;
            })

        // Debug helpers
        .def("get_debug_summary", &OccupancyMap::get_debug_summary,
             py::arg("vehicle_depth"))

        .def("get_grid_snapshot",
            [](const OccupancyMap& self) -> py::dict {
                py::dict d;
                d["grid"]    = self.grid();   // Eigen → numpy (copy)
                d["nx"]      = self.nx();
                d["nz"]      = self.nz();
                d["cx"]      = self.cx();
                d["dx"]      = self.cfg().dx;
                d["dz"]      = self.cfg().dz;
                d["grid_origin_x"]          = self.grid_origin_x();
                d["manifold_grid_origin_x"] = self.manifold_grid_origin_x();
                d["z_min"]   = self.grid_origin_z();
                d["z_max"]   = self.grid_origin_z() + self.nz() * self.cfg().dz;
                d["manifold_iz"] = self.manifold_iz();
                d["manifold_z"]  = self.manifold_z();
                d["cmd_depth"]   = self.cmd_depth();

                py::list wps;
                for (auto& [x, z] : self.path_waypoints())
                    wps.append(py::make_tuple(x, z));
                d["path_waypoints"] = wps;

                double alt = self.dvl_altitude();
                if (std::isnan(alt)) {
                    d["dvl_altitude"] = py::none();
                } else {
                    d["dvl_altitude"] = py::float_(alt);
                }
                d["control_mode"] = self.control_mode();
                return d;
            })
    ;

    // -----------------------------------------------------------------------
    // ObstacleMapper
    // -----------------------------------------------------------------------
    py::class_<ObstacleMapper>(m, "ObstacleMapper")
        .def(py::init<const OccupancyMapConfig&, const DVLConfig&,
                      const SonarConfig&, const AltimeterConfig&>(),
             py::arg("config"),
             py::arg("dvl_config"),
             py::arg("sonar_config"),
             py::arg("altimeter_config") = AltimeterConfig{})

        .def("reset", &ObstacleMapper::reset, py::arg("pose"))

        // Three update_sensor overloads dispatched by SensorType + measurement type
        .def("update_sensor",
            [](ObstacleMapper& self, SensorType t, const DVLMeasurement& m, const Pose& p) {
                self.update_sensor(t, m, p);
            },
            py::arg("sensor_type"), py::arg("measurement"), py::arg("pose"))

        .def("update_sensor",
            [](ObstacleMapper& self, SensorType t, const AltimeterMeasurement& m, const Pose& p) {
                self.update_sensor(t, m, p);
            },
            py::arg("sensor_type"), py::arg("measurement"), py::arg("pose"))

        .def("update_sensor",
            [](ObstacleMapper& self, SensorType t, const SonarMeasurement& m, const Pose& p) {
                self.update_sensor(t, m, p);
            },
            py::arg("sensor_type"), py::arg("measurement"), py::arg("pose"))

        .def("update_pose",  &ObstacleMapper::update_pose,  py::arg("pose"))
        .def("get_control",  &ObstacleMapper::get_control)
        .def("get_altitude", &ObstacleMapper::get_altitude)

        .def_property_readonly("omap",
            [](ObstacleMapper& self) -> OccupancyMap& { return self.omap(); },
            py::return_value_policy::reference_internal)
    ;
}
