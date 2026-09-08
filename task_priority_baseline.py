"""Set-based task-priority controller — a research baseline, not deployed code.

This exists to answer one question for the publication: does obstacle avoidance
on this platform actually require *commitment*, or would a well-posed reactive
controller do?  It is deliberately memoryless, and it is the thing the latch
controller is compared against.

It is NOT an alternative implementation of the deployed algorithm.  The C++ core
remains the single implementation of that; this lives in Python, is used only by
the simulator and the comparison harness, and must never be vendored into
acfr-lcm.

Design, and why it is a fair comparison rather than a straw man
--------------------------------------------------------------
The controller **shares the deployed perception stack**.  It wraps a real
``occupancy_map_cpp.ObstacleMapper``, feeds it exactly the same measurements,
and reads the same occupancy grid and cliff manifold the latch controller reads.
Only the arbitration differs.  Anything the comparison shows is therefore a
property of the control law, not of the mapping.

Three tasks in strict priority order, resolved in the single vertical DOF:

  1. Safety clearance   (set-based)  keep terrain below the hull further than
                                     ``safety_below_m``
  2. Forward clearance  (set-based)  stay above ``imaging_altitude`` over the
                                     shallowest terrain within ``cliff_standoff``
                                     ahead of the nose
  3. Imaging altitude   (equality)   drive altitude to ``imaging_altitude``

With one degree of freedom the null-space projector of an active task is
identically zero — a scalar task with a non-zero Jacobian leaves no null space
for lower-priority tasks to act in.  The null-space-based formulation therefore
reduces exactly to strict priority arbitration here, and that is what is
implemented.  This is a reduction of NSB, not a simplification of it; the
distinction matters if the comparison is ever extended to a second DOF, where
the projectors stop being trivial and the tasks genuinely compose.

Set-based tasks activate on constraint violation and deactivate once the
constraint is satisfied with margin (``buffer_m``), following the usual
treatment of inequality tasks.  Note carefully what that hysteresis is and is
not: it is a band on the *currently measured* constraint value, which prevents
chatter at the boundary.  It is not memory of an obstacle that has stopped being
observable.  When the peak leaves the forward window the task has nothing to
measure, the constraint reads as satisfied, and the task deactivates — which is
the behaviour under test.

``dwell_s`` optionally holds an activated set-based task for a minimum duration.
That is the repair: it reintroduces commitment inside the task-priority
formalism, and the three-way comparison (latch / plain / dwell) is what shows
the commitment mechanism is necessary rather than stylistic.
"""

import numpy as np
import occupancy_map_cpp as cpp

__all__ = ["TaskPriorityMapper", "TPControl"]


class TPControl:
    """Mirrors occupancy_map_cpp.ControlCommand so the simulator can consume it."""

    __slots__ = ("vx", "vertical_mode", "vertical_target")

    def __init__(self, vx, vertical_mode, vertical_target):
        self.vx = vx
        self.vertical_mode = vertical_mode
        self.vertical_target = vertical_target


class TaskPriorityMapper:
    """Drop-in replacement for ObstacleMapper that arbitrates instead of committing.

    Exposes the same surface the simulator and visualizers use: ``reset``,
    ``update_sensor``, ``update_pose``, ``get_altitude``, ``get_control`` and
    ``omap``.  ``omap`` is the wrapped C++ map, so grid and manifold rendering
    keeps working unchanged.

    Args:
        cfg, dvl_cfg, sonar_cfg, alt_cfg: same configs as ObstacleMapper.
        buffer_m: margin by which a set-based constraint must be satisfied
                  before its task deactivates (chatter suppression).
        dwell_s:  minimum time an activated set-based task stays active.  0.0
                  gives the plain memoryless baseline; a positive value gives
                  the commitment-repaired variant.
    """

    def __init__(self, cfg, dvl_cfg, sonar_cfg, alt_cfg=None,
                 buffer_m=0.25, dwell_s=0.0):
        self._inner = cpp.ObstacleMapper(
            cfg, dvl_cfg, sonar_cfg, alt_cfg or cpp.AltimeterConfig())
        self.cfg = cfg
        self.buffer_m = float(buffer_m)
        self.dwell_s = float(dwell_s)

        self._t = 0.0
        self._depth = 0.0
        self._active = {1: False, 2: False}     # set-based task activation
        self._activated_at = {1: -1e9, 2: -1e9}
        self.control_mode = "ALT_FOLLOW"        # reported for logging/plots
        self.active_task = 3
        self.transitions = 0

    # ---- ObstacleMapper surface -------------------------------------------

    @property
    def omap(self):
        return self._inner.omap

    def reset(self, pose):
        self._inner.reset(pose)
        self._depth = pose.depth
        self._active = {1: False, 2: False}
        self._activated_at = {1: -1e9, 2: -1e9}
        self.control_mode = "ALT_FOLLOW"
        self.active_task = 3
        self.transitions = 0

    def update_sensor(self, sensor_type, measurement, pose):
        self._inner.update_sensor(sensor_type, measurement, pose)
        self._depth = pose.depth

    def update_pose(self, pose):
        self._inner.update_pose(pose)
        self._depth = pose.depth

    def set_altimeter_altitude(self, v):
        self._inner.set_altimeter_altitude(v)

    def get_altitude(self):
        return self._inner.get_altitude()

    def tick(self, dt):
        """Advance the controller clock.  Only needed when dwell_s > 0."""
        self._t += dt

    # ---- task evaluation ---------------------------------------------------

    def _manifold_world(self):
        """(world_x, terrain_depth, observed) arrays for the current manifold."""
        omap = self._inner.omap
        z = np.asarray(omap.manifold_z, dtype=float)
        obs = np.asarray(omap.manifold_observed, dtype=bool)
        x0 = omap.manifold_grid_origin_x
        x = x0 + np.arange(len(z)) * self.cfg.dx
        return x, z, obs

    def _sigma_safety(self):
        """Clearance under the hull, minus the required minimum.  Negative = violated."""
        c = self.cfg
        omap = self._inner.omap
        x, z, obs = self._manifold_world()
        v_x = omap.grid_to_world_x(omap.cx)
        half = c.vehicle_length / 2.0
        band = obs & (x >= v_x - half - c.safety_standoff_m) & (x <= v_x + half)
        if not band.any():
            return np.inf
        shallowest = float(np.min(z[band]))          # smallest depth = highest terrain
        clearance = shallowest - self._depth         # terrain below the hull
        return clearance - c.safety_below_m

    def _sigma_forward(self):
        """Depth error against the forward terrain requirement.  Negative = violated.

        Recomputed from the current window every cycle.  Nothing is remembered:
        once the peak is behind the nose it stops contributing, which is the
        whole point of the baseline.
        """
        c = self.cfg
        omap = self._inner.omap
        x, z, obs = self._manifold_world()
        v_x = omap.grid_to_world_x(omap.cx)
        nose = v_x + c.vehicle_length / 2.0
        win = obs & (x >= nose) & (x <= nose + c.cliff_standoff)
        if not win.any():
            return np.inf, np.nan
        peak_z = float(np.min(z[win]))
        required_depth = peak_z - c.imaging_altitude
        return required_depth - self._depth, required_depth

    def _set_based(self, task_id, sigma):
        """Activation with a satisfaction buffer and optional dwell."""
        if not np.isfinite(sigma):
            violated, satisfied = False, True
        else:
            violated = sigma < 0.0
            satisfied = sigma > self.buffer_m
        if self._active[task_id]:
            held = (self._t - self._activated_at[task_id]) < self.dwell_s
            if satisfied and not held:
                self._active[task_id] = False
        elif violated:
            self._active[task_id] = True
            self._activated_at[task_id] = self._t
        return self._active[task_id]

    # ---- arbitration -------------------------------------------------------

    def get_control(self):
        c = self.cfg
        prev = self.active_task

        s1 = self._sigma_safety()
        s2, required_depth = self._sigma_forward()

        t1 = self._set_based(1, s1)
        t2 = self._set_based(2, s2)

        if t1:
            # Ascend until the hull clearance constraint is satisfied again.
            deficit = c.safety_below_m - (s1 + c.safety_below_m)
            target = max(0.0, self._depth - max(deficit, 0.0) - self.buffer_m)
            mode, vtarget, task = "DEPTH_HOLD", target, 1
        elif t2:
            target = required_depth if np.isfinite(required_depth) else self._depth
            mode, vtarget, task = "DEPTH_HOLD", target, 2
        else:
            mode, vtarget, task = "ALT_FOLLOW", c.imaging_altitude, 3

        # Forward speed policy mirrors the latch controller so the comparison is
        # about commitment, not about how fast either one flies: hold station
        # while a large in-place ascent is outstanding.
        if task in (1, 2) and (self._depth - vtarget) > c.altitude_overshoot_threshold_m:
            vx = 0.0
        else:
            vx = c.survey_speed

        self.active_task = task
        self.control_mode = {1: "SAFETY_CLEAR", 2: "FWD_CLEAR", 3: "ALT_FOLLOW"}[task]
        if task != prev:
            self.transitions += 1
        return TPControl(vx, mode, vtarget)
