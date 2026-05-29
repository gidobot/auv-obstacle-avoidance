#!/usr/bin/env python3
"""
LCM log playback visualizer for AUV obstacle avoidance testing.

Reads an LCM log file from a real AUV mission and feeds the sensor data
(DVL bottom-track, altimeter, forward sonar, and navigation) into the
ObstacleMapper, displaying the resulting occupancy grid and avoidance
manifold in the same browser-based visualization used by the simulator.

Usage:
    python lcm_playback_visualizer.py /path/to/logfile.lcm
    python lcm_playback_visualizer.py /path/to/logfile.lcm --vehicle DURHAM
    python lcm_playback_visualizer.py /path/to/logfile.lcm --speed 4

LCM topics consumed (vehicle name auto-detected from *.ACFR_NAV channel):
    <VEHICLE>.ACFR_NAV             navigation solution (pose)
    <VEHICLE>.NUCLEUS.ALTIMETER    downward altimeter range
    <VEHICLE>.NUCLEUS.BOTTOMTRACK  DVL 3-beam bottom-track
    <VEHICLE>.ISA500_FWD           forward-looking sonar

The ObstacleMapper uses NED frame: nav.x = north (m), nav.y = east (m).
If the vehicle uses a different convention (e.g. x = east), swap with --swap-xy.
"""

import asyncio
import http.server
import json
import os
import sys
import threading
import time
import argparse
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# LCM types path — niceauv pipx venv ships lcm + acfrlcm + senlcm for Python 3.12
# ---------------------------------------------------------------------------
_DEFAULT_LCM_TYPES_PATH = (
    '/home/gidobot/.local/share/pipx/venvs/niceauv/lib/python3.12/site-packages'
)


def _ensure_lcm_path(path: str) -> None:
    if path not in sys.path:
        sys.path.insert(0, path)


# ---------------------------------------------------------------------------
# ObstacleMapper imports
# ---------------------------------------------------------------------------
def _safe(v) -> Optional[float]:
    """Return float v, or None if NaN/inf (so json.dumps emits null not NaN)."""
    if v is None:
        return None
    try:
        f = float(v)
        return None if (f != f or f == float('inf') or f == float('-inf')) else f
    except (TypeError, ValueError):
        return None


from occupancy_map import (
    ObstacleMapper, OccupancyMapConfig,
    DVLConfig, SonarConfig, AltimeterConfig,
    Pose, SensorType,
    DVLMeasurement, AltimeterMeasurement, SonarMeasurement,
)

# ---------------------------------------------------------------------------
# Browser HTML client — reuse the 3D visualizer's client unchanged
# ---------------------------------------------------------------------------
try:
    from visualizer import HTML_CLIENT_3D
except ImportError:
    HTML_CLIENT_3D = "<html><body>visualizer.py not found</body></html>"

try:
    import websockets
except ImportError:
    raise SystemExit("websockets package not found: pip install websockets")


# ---------------------------------------------------------------------------
# LCM event loading
# ---------------------------------------------------------------------------

_CHANNEL_SUFFIXES = (
    'ACFR_NAV',
    'NUCLEUS.ALTIMETER',
    'NUCLEUS.BOTTOMTRACK',
    'ISA500_FWD',
)


def detect_vehicle_name(log_path: str) -> Optional[str]:
    """Scan a log file and return the vehicle name from the first *.ACFR_NAV channel."""
    import lcm
    log = lcm.EventLog(log_path, 'r')
    for event in log:
        if event.channel.endswith('.ACFR_NAV'):
            return event.channel[: -len('.ACFR_NAV')]
    return None


def load_events(log_path: str, vehicle_name: str) -> List[tuple]:
    """
    Load all relevant events from the log, sorted by timestamp.

    Returns a list of (utime_us, suffix, raw_bytes) where suffix is one of
    the _CHANNEL_SUFFIXES strings.
    """
    import lcm
    channels = {f"{vehicle_name}.{s}": s for s in _CHANNEL_SUFFIXES}
    events = []
    log = lcm.EventLog(log_path, 'r')
    count = 0
    for event in log:
        suffix = channels.get(event.channel)
        if suffix is not None:
            events.append((event.timestamp, suffix, event.data))
        count += 1
        if count % 50000 == 0:
            print(f"  scanned {count} log events, collected {len(events)}...", end='\r')
    print(f"  scanned {count} log events, collected {len(events)} matching messages")
    events.sort(key=lambda e: e[0])
    return events


# ---------------------------------------------------------------------------
# DVL beam hit positions in world XY (for top-down map overlay)
# ---------------------------------------------------------------------------

def _dvl_hit_xy(nav_x: float, nav_y: float, heading_rad: float,
                ranges: np.ndarray, dvl_cfg: DVLConfig,
                hit_surface: np.ndarray) -> list:
    """Return list of [wx, wy] or None for each reported DVL beam's surface hit."""
    hits = []
    dirs = dvl_cfg.beam_directions_3d  # shape (n_cfg, 3): [fwd, stbd, down]
    n = min(len(ranges), len(dirs))    # guard against nbeams < configured beams
    cos_h = np.cos(heading_rad)
    sin_h = np.sin(heading_rad)
    for i in range(n):
        if not hit_surface[i] or ranges[i] <= 0:
            hits.append(None)
            continue
        fwd, stbd = dirs[i, 0], dirs[i, 1]
        r = ranges[i]
        dx_fwd = r * fwd
        dx_stbd = r * stbd
        wx = nav_x + dx_fwd * cos_h - dx_stbd * sin_h
        wy = nav_y + dx_fwd * sin_h + dx_stbd * cos_h
        hits.append([float(wx), float(wy)])
    return hits


def _sonar_hit_xy(nav_x: float, nav_y: float, heading_rad: float,
                  range_m: float, vehicle_length: float,
                  hit: bool) -> Optional[list]:
    """Return [wx, wy] of the forward sonar return, or None."""
    if not hit or range_m <= 0:
        return None
    nose_offset = vehicle_length / 2.0
    total = nose_offset + range_m
    cos_h = np.cos(heading_rad)
    sin_h = np.sin(heading_rad)
    return [float(nav_x + total * cos_h), float(nav_y + total * sin_h)]


# ---------------------------------------------------------------------------
# PlaybackServer
# ---------------------------------------------------------------------------

class PlaybackServer:
    """WebSocket + HTTP server for LCM log playback visualization."""

    def __init__(
        self,
        events: List[tuple],
        vehicle_name: str,
        log_path: str,
        http_port: int = 8082,
        ws_port: int = 8083,
        initial_speed: float = 1.0,
        swap_xy: bool = False,
        lcm_types_path: str = _DEFAULT_LCM_TYPES_PATH,
    ):
        self.events = events
        self.vehicle_name = vehicle_name
        self.log_path = log_path
        self.http_port = http_port
        self.ws_port = ws_port
        self.swap_xy = swap_xy
        self.lcm_types_path = lcm_types_path

        # Playback state
        self.playing = False
        self.time_accel = initial_speed
        self._event_idx = 0
        self._log_start_utime: Optional[int] = events[0][0] if events else None
        self._play_start_wall: Optional[float] = None
        self._play_start_log_utime: Optional[int] = None
        self._elapsed_log_us: float = 0.0   # accumulated log-time when paused

        # Clients
        self.clients: set = set()

        # Sensor configs — Nortek Nucleus 1000 3-beam + ISA500 defaults
        self._dvl_cfg = DVLConfig()
        self._sonar_cfg = SonarConfig()
        self._alt_cfg = AltimeterConfig()

        # ObstacleMapper
        omap_cfg = OccupancyMapConfig()
        self.mapper = ObstacleMapper(omap_cfg, self._dvl_cfg,
                                     self._sonar_cfg, self._alt_cfg)

        # Tracked vehicle state (updated from ACFR_NAV)
        self._nav_x: float = 0.0
        self._nav_y: float = 0.0
        self._nav_depth: float = 0.0
        self._nav_heading: float = 0.0
        self._nav_altitude: float = np.nan
        self._initialized: bool = False

        # Visualization state
        self._arc_local: float = 0.0    # accumulated along-track distance
        self._xy_trail: list = []
        self._dvl_hit_xy: list = []
        self._sonar_hit_xy: Optional[list] = None
        self._elapsed_s: float = 0.0

        # LCM decoders (loaded lazily after path is set)
        self._nav_t = None
        self._alt_t = None
        self._btk_t = None
        self._isa_t = None

        # Pre-build terrain map from nav trajectory bounds
        self._terrain_map_msg = self._build_terrain_map_from_events()

    def _load_decoders(self) -> None:
        _ensure_lcm_path(self.lcm_types_path)
        from acfrlcm import auv_acfr_nav_t
        from senlcm import nucleus_altimeter_t, nucleus_bottomtrack_t, isa500_t
        self._nav_t = auv_acfr_nav_t
        self._alt_t = nucleus_altimeter_t
        self._btk_t = nucleus_bottomtrack_t
        self._isa_t = isa500_t

    def _build_terrain_map_from_events(self) -> str:
        """Build a flat terrain map sized to cover the vehicle's actual trajectory."""
        _ensure_lcm_path(self.lcm_types_path)
        from acfrlcm import auv_acfr_nav_t

        xs, ys, depths = [], [], []
        for _utime, suffix, raw in self.events:
            if suffix == 'ACFR_NAV':
                try:
                    msg = auv_acfr_nav_t.decode(raw)
                    north, east = (msg.y, msg.x) if self.swap_xy else (msg.x, msg.y)
                    xs.append(north)
                    ys.append(east)
                    if msg.depth > 0:
                        depths.append(msg.depth)
                except Exception:
                    pass

        margin = 30.0
        dx = dy = 2.0
        if xs:
            ox = min(xs) - margin
            oy = min(ys) - margin
            nx = max(4, int(np.ceil((max(xs) + margin - ox) / dx)))
            ny = max(4, int(np.ceil((max(ys) + margin - oy) / dy)))
        else:
            ox, oy, nx, ny = -120.0, -120.0, 120, 120

        # Estimate operating depth from nav altitude field; fall back to 20m
        avg_depth = float(np.mean(depths)) if depths else 20.0
        min_z = max(0.0, avg_depth - 5.0)
        max_z = avg_depth + 5.0

        data = [avg_depth] * (nx * ny)
        return json.dumps({
            'type': 'terrain_map',
            'nx': nx, 'ny': ny,
            'dx': dx, 'dy': dy,
            'ox': ox, 'oy': oy,
            'minZ': min_z, 'maxZ': max_z,
            'data': data,
            'mission_path': [],
        })

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------

    def _nav_to_pose(self, msg) -> Pose:
        if self.swap_xy:
            north, east = msg.y, msg.x
        else:
            north, east = msg.x, msg.y
        return Pose(north=north, east=east,
                    depth=msg.depth, heading=msg.heading)

    # ------------------------------------------------------------------
    # Event processing
    # ------------------------------------------------------------------

    def _process_event(self, suffix: str, raw: bytes) -> None:
        """Decode one LCM message and feed it into the mapper."""
        cfg = self.mapper.omap.cfg

        if suffix == 'ACFR_NAV':
            msg = self._nav_t.decode(raw)
            pose = self._nav_to_pose(msg)

            if not self._initialized:
                self.mapper.reset(pose)
                self._initialized = True
                self._nav_x = pose.north
                self._nav_y = pose.east
                self._nav_depth = pose.depth
                self._nav_heading = pose.heading
                self._nav_altitude = getattr(msg, 'altitude', np.nan)
                return

            # Accumulate along-track distance
            dn = pose.north - self._nav_x
            de = pose.east - self._nav_y
            cos_h = np.cos(self._nav_heading)
            sin_h = np.sin(self._nav_heading)
            ds = dn * cos_h + de * sin_h
            if ds > 0:
                self._arc_local += ds

            self._nav_x = pose.north
            self._nav_y = pose.east
            self._nav_depth = pose.depth
            self._nav_heading = pose.heading
            self._nav_altitude = getattr(msg, 'altitude', np.nan)

            self.mapper.update_pose(pose)
            self._xy_trail.append([float(pose.north), float(pose.east)])
            if len(self._xy_trail) > 2000:
                self._xy_trail = self._xy_trail[-2000:]

        elif suffix == 'NUCLEUS.BOTTOMTRACK' and self._initialized:
            msg = self._btk_t.decode(raw)
            # distance_beam is a fixed 3-element array in the LCM type; nbeams
            # is unreliable (often 0) in this ACFR driver configuration.
            n = min(len(msg.distance_beam), len(self._dvl_cfg.beams))
            ranges = np.array(msg.distance_beam[:n], dtype=float)
            valid = np.array(msg.distance_beam_valid[:n], dtype=bool)
            # Sentinel: distance == 0.0 means invalid even if flag not set
            valid &= ranges > 0.0
            pose = Pose(north=self._nav_x, east=self._nav_y,
                        depth=self._nav_depth, heading=self._nav_heading)
            self.mapper.update_sensor(
                SensorType.DVL,
                DVLMeasurement(ranges=ranges, hit_surface=valid),
                pose,
            )
            self._dvl_hit_xy = _dvl_hit_xy(
                self._nav_x, self._nav_y, self._nav_heading,
                ranges, self._dvl_cfg, valid,
            )

        elif suffix == 'NUCLEUS.ALTIMETER' and self._initialized:
            msg = self._alt_t.decode(raw)
            dist = msg.altimeter_distance
            hit = dist > 0.0 and dist < self._alt_cfg.max_range - 0.05
            pose = Pose(north=self._nav_x, east=self._nav_y,
                        depth=self._nav_depth, heading=self._nav_heading)
            self.mapper.update_sensor(
                SensorType.ALTIMETER,
                AltimeterMeasurement(range_m=dist if dist > 0 else 1.0, hit=hit),
                pose,
            )

        elif suffix == 'ISA500_FWD' and self._initialized:
            msg = self._isa_t.decode(raw)
            dist = msg.distance
            max_r = self._sonar_cfg.max_range
            hit = 0.0 < dist < max_r - 0.1
            pose = Pose(north=self._nav_x, east=self._nav_y,
                        depth=self._nav_depth, heading=self._nav_heading)
            self.mapper.update_sensor(
                SensorType.SONAR,
                SonarMeasurement(range_m=dist if dist > 0 else max_r, hit=hit),
                pose,
            )
            self._sonar_hit_xy = _sonar_hit_xy(
                self._nav_x, self._nav_y, self._nav_heading,
                dist, self.mapper.omap.cfg.vehicle_length, hit,
            )

    def _current_log_utime(self) -> int:
        """Return the log-time (μs) we should have processed up to right now."""
        if not self.playing or self._play_start_wall is None:
            return self._log_start_utime + int(self._elapsed_log_us)
        wall_elapsed = time.monotonic() - self._play_start_wall
        return self._play_start_log_utime + int(
            wall_elapsed * self.time_accel * 1e6
        )

    def _pump_events(self) -> None:
        """Process all queued events up to the current playback clock."""
        target_utime = self._current_log_utime()
        while self._event_idx < len(self.events):
            utime, suffix, raw = self.events[self._event_idx]
            if utime > target_utime:
                break
            self._process_event(suffix, raw)
            self._event_idx += 1

    def _elapsed_log_s(self) -> float:
        """Elapsed log-time (s) corresponding to the current cursor position."""
        if self._log_start_utime is None or not self.events:
            return 0.0
        if self._event_idx == 0:
            return 0.0
        # Use last-processed event's timestamp
        idx = min(self._event_idx, len(self.events) - 1)
        return (self.events[idx][0] - self._log_start_utime) / 1e6

    # ------------------------------------------------------------------
    # State message
    # ------------------------------------------------------------------

    def _build_state_msg(self) -> str:
        omap = self.mapper.omap
        cfg = omap.cfg
        snap = omap.get_grid_snapshot()

        manifold_z = [None if np.isnan(z) else float(z)
                      for z in snap['manifold_z']]

        # Flat terrain profile at estimated seafloor depth
        alt = self.mapper.get_altitude()
        seafloor_z = (self._nav_depth + alt
                      if not np.isnan(alt) else self._nav_depth + cfg.imaging_altitude)
        vx = self._arc_local
        n_pts = 40
        terrain_profile = [
            [float(vx - cfg.horizon_back + i * (cfg.horizon_fwd + cfg.horizon_back) / (n_pts - 1)),
             float(seafloor_z)]
            for i in range(n_pts)
        ]

        state = {
            'sim_mode': '3d',
            'backend': 'python',
            'vehicle_x': float(self._arc_local),
            'vehicle_wx': float(self._nav_x),
            'vehicle_y': float(self._nav_y),
            'vehicle_z': float(self._nav_depth),
            'vehicle_heading': float(self._nav_heading),
            'terrain_z': float(seafloor_z),
            'altitude': float(alt) if not np.isnan(alt) else None,
            'time': float(self._elapsed_log_s()),
            'cmd_depth': _safe(omap.get_commanded_depth_at_vehicle()),
            'dvl_altitude': snap['dvl_altitude'],
            'control_mode': snap['control_mode'],
            'terrain_label': f'LCM: {os.path.basename(self.log_path)} / {self.vehicle_name}',
            'grid': snap['grid'].flatten().tolist(),
            'nx': snap['nx'], 'nz': snap['nz'],
            'dx': snap['dx'], 'dz': snap['dz'],
            'cx': snap['cx'],
            'grid_origin_x': float(snap['grid_origin_x']),
            'manifold_grid_origin_x': float(snap['manifold_grid_origin_x']),
            'z_min': float(snap['z_min']), 'z_max': float(snap['z_max']),
            'horizon_fwd': float(cfg.horizon_fwd),
            'horizon_back': float(cfg.horizon_back),
            'vehicle_length': float(cfg.vehicle_length),
            'manifold_z': manifold_z,
            'cmd_depth_profile': [float(d) if not np.isnan(d) else None
                                  for d in snap['cmd_depth']],
            'path_waypoints': snap['path_waypoints'],
            'terrain_profile': terrain_profile,
            'xy_trail': self._xy_trail[-500:],
            'dvl_hit_xy': self._dvl_hit_xy,
            'sonar_hit_xy': self._sonar_hit_xy,
            'enable_dvl': True,
            'enable_altimeter': True,
            'enable_sonar': True,
        }
        return json.dumps(state)

    # ------------------------------------------------------------------
    # Async server
    # ------------------------------------------------------------------

    async def _playback_loop(self) -> None:
        dt = 0.05   # 20 Hz broadcast
        while True:
            if self.playing and self._event_idx < len(self.events):
                self._pump_events()
                if self._event_idx >= len(self.events):
                    # Reached end of log
                    self.playing = False
                    print("\nEnd of log reached.")

            if self.clients:
                msg = self._build_state_msg()
                dead = set()
                for client in list(self.clients):
                    try:
                        await client.send(msg)
                    except websockets.exceptions.ConnectionClosed:
                        dead.add(client)
                self.clients -= dead

            await asyncio.sleep(dt)

    async def _ws_handler(self, websocket) -> None:
        self.clients.add(websocket)
        try:
            await websocket.send(self._terrain_map_msg)
            async for message in websocket:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue
                cmd = data.get('cmd')
                if cmd == 'play':
                    if not self.playing:
                        # Resume: record where we are in log-time
                        self._play_start_wall = time.monotonic()
                        self._play_start_log_utime = (
                            self._log_start_utime + int(self._elapsed_log_us)
                        )
                        self.playing = True
                elif cmd == 'pause':
                    if self.playing:
                        self._elapsed_log_us = (
                            self._current_log_utime() - self._log_start_utime
                        )
                        self.playing = False
                elif cmd == 'reset':
                    self._event_idx = 0
                    self._elapsed_log_us = 0.0
                    self._play_start_wall = None
                    self._play_start_log_utime = None
                    self.playing = False
                    self._initialized = False
                    self._arc_local = 0.0
                    self._xy_trail = []
                    self._dvl_hit_xy = []
                    self._sonar_hit_xy = None
                    omap_cfg = OccupancyMapConfig()
                    self.mapper = ObstacleMapper(omap_cfg, self._dvl_cfg,
                                                 self._sonar_cfg, self._alt_cfg)
                    await websocket.send(self._terrain_map_msg)
                elif cmd == 'param':
                    key = data.get('key')
                    val = data.get('value')
                    if key == 'time_accel':
                        if self.playing:
                            self._elapsed_log_us = (
                                self._current_log_utime() - self._log_start_utime
                            )
                            self._play_start_wall = time.monotonic()
                            self._play_start_log_utime = (
                                self._log_start_utime + int(self._elapsed_log_us)
                            )
                        self.time_accel = float(val)
                    elif key is not None and hasattr(self.mapper.omap.cfg, key):
                        setattr(self.mapper.omap.cfg, key, float(val))
                # Ignore 'configure' commands (no terrain/trajectory to reconfigure)
        finally:
            self.clients.discard(websocket)

    async def start(self) -> None:
        """Start HTTP and WebSocket servers, then run the playback loop."""
        self._load_decoders()

        # Patch the HTML client for LCM playback mode.
        client_html = (
            HTML_CLIENT_3D
            # Fix hardcoded WS port
            .replace("ws://localhost:8081",
                     f"ws://localhost:{self.ws_port}")
            # Null-safe altitude / cmd_depth stats
            .replace("'Alt: ' + s.altitude.toFixed(2) + 'm'",
                     "(s.altitude != null ? 'Alt: ' + s.altitude.toFixed(2) + 'm' : 'Alt: --')")
            .replace("'Cmd: ' + s.cmd_depth.toFixed(2) + 'm'",
                     "(s.cmd_depth != null ? 'Cmd: ' + s.cmd_depth.toFixed(2) + 'm' : 'Cmd: --')")
            # Top-down view: keep vehicle centered (replace fixed terrain-origin
            # coordinate system with a vehicle-centred ±60 m window)
            .replace(
                "  // Draw terrain background\n"
                "  if (terrainImageData) ctx.drawImage(terrainImageData, 0, 0);\n"
                "\n"
                "  const { nx, ny, ox, oy, dx, dy } = terrainMap;\n"
                "  const worldW = nx * dx, worldH = ny * dy;\n"
                "\n"
                "  // World → pixel\n"
                "  function toPixel(wx, wy) {\n"
                "    return [\n"
                "      (wx - ox) / worldW * mapW,\n"
                "      (1 - (wy - oy) / worldH) * mapH,   // north up\n"
                "    ];\n"
                "  }",
                "  const { nx, ny, ox, oy, dx, dy } = terrainMap;\n"
                "  const worldW = nx * dx, worldH = ny * dy;\n"
                "\n"
                "  // Vehicle-centred view: show ±viewHalf metres around the vehicle\n"
                "  const viewHalf = 60;\n"
                "  const vwxC = (s.vehicle_wx !== undefined) ? s.vehicle_wx : s.vehicle_x;\n"
                "  const vyC  = s.vehicle_y || 0;\n"
                "  const viewOx = vwxC - viewHalf, viewOy = vyC - viewHalf;\n"
                "  const viewSize = viewHalf * 2;\n"
                "\n"
                "  // Draw terrain background sliced to the centred window\n"
                "  ctx.fillStyle = '#1a1a1a'; ctx.fillRect(0, 0, mapW, mapH);\n"
                "  if (terrainImageData) {\n"
                "    const srcX = (viewOx - ox) / worldW * mapW;\n"
                "    const srcY = (1 - (viewOy + viewSize - oy) / worldH) * mapH;\n"
                "    const srcW = viewSize / worldW * mapW;\n"
                "    const srcH = viewSize / worldH * mapH;\n"
                "    ctx.drawImage(terrainImageData, srcX, srcY, srcW, srcH, 0, 0, mapW, mapH);\n"
                "  }\n"
                "\n"
                "  // World → pixel (vehicle-centred)\n"
                "  function toPixel(wx, wy) {\n"
                "    return [\n"
                "      (wx - viewOx) / viewSize * mapW,\n"
                "      (1 - (wy - viewOy) / viewSize) * mapH,  // north up\n"
                "    ];\n"
                "  }"
            )
            # Update grid-line loop bounds to use the centred view window
            .replace(
                "  const gx0 = Math.ceil(ox / 20) * 20;\n"
                "  for (let gx = gx0; gx <= ox + worldW; gx += 20) {\n"
                "    const [px] = toPixel(gx, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, mapH); ctx.stroke();\n"
                "  }\n"
                "  const gy0 = Math.ceil(oy / 20) * 20;\n"
                "  for (let gy = gy0; gy <= oy + worldH; gy += 20) {\n"
                "    const [, py] = toPixel(0, gy); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(mapW, py); ctx.stroke();\n"
                "  }",
                "  const gx0 = Math.ceil(viewOx / 20) * 20;\n"
                "  for (let gx = gx0; gx <= viewOx + viewSize; gx += 20) {\n"
                "    const [px] = toPixel(gx, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, mapH); ctx.stroke();\n"
                "  }\n"
                "  const gy0 = Math.ceil(viewOy / 20) * 20;\n"
                "  for (let gy = gy0; gy <= viewOy + viewSize; gy += 20) {\n"
                "    const [, py] = toPixel(0, gy); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(mapW, py); ctx.stroke();\n"
                "  }"
            )
        ).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self_):
                self_.send_response(200)
                self_.send_header('Content-Type', 'text/html')
                self_.end_headers()
                self_.wfile.write(client_html)

            def log_message(self_, fmt, *args):
                pass

        http_thread = threading.Thread(
            target=lambda: http.server.HTTPServer(
                ('0.0.0.0', self.http_port), Handler
            ).serve_forever(),
            daemon=True,
        )
        http_thread.start()

        n_events = len(self.events)
        log_dur_s = (
            (self.events[-1][0] - self.events[0][0]) / 1e6
            if n_events >= 2 else 0.0
        )
        print(f"Vehicle:    {self.vehicle_name}")
        print(f"Log:        {self.log_path}")
        print(f"Events:     {n_events}  ({log_dur_s:.1f} s of data)")
        print(f"HTTP:       http://localhost:{self.http_port}")
        print(f"WebSocket:  ws://localhost:{self.ws_port}")
        print("Open the URL above in a browser, then press Play.")

        async with websockets.serve(self._ws_handler, '0.0.0.0', self.ws_port):
            await self._playback_loop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='LCM log playback visualizer for AUV obstacle avoidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('log', metavar='LOG_FILE',
                        help='Path to the LCM log file')
    parser.add_argument('--vehicle', metavar='NAME',
                        help='Vehicle name (e.g. DURHAM); auto-detected if omitted')
    parser.add_argument('--speed', type=float, default=1.0, metavar='X',
                        help='Initial playback speed multiplier (default: 1.0)')
    parser.add_argument('--http-port', type=int, default=8082, metavar='PORT',
                        help='HTTP port for browser client (default: 8082)')
    parser.add_argument('--ws-port', type=int, default=8083, metavar='PORT',
                        help='WebSocket port (default: 8083)')
    parser.add_argument('--swap-xy', action='store_true',
                        help='Swap nav.x/nav.y if vehicle uses east-first convention')
    parser.add_argument('--lcm-types-path', default=_DEFAULT_LCM_TYPES_PATH,
                        metavar='PATH',
                        help='Path to directory containing perls/lcmtypes package')
    args = parser.parse_args()

    if not os.path.isfile(args.log):
        parser.error(f"Log file not found: {args.log}")

    _ensure_lcm_path(args.lcm_types_path)

    try:
        import lcm  # noqa: F401
    except ImportError:
        raise SystemExit("lcm package not found — install the LCM Python bindings")

    vehicle = args.vehicle
    if vehicle is None:
        print(f"Scanning {args.log} for vehicle name...")
        vehicle = detect_vehicle_name(args.log)
        if vehicle is None:
            raise SystemExit("Could not detect vehicle name. Use --vehicle NAME.")
        print(f"Detected vehicle: {vehicle}")

    print(f"Loading events from {args.log}...")
    events = load_events(args.log, vehicle)
    if not events:
        raise SystemExit(f"No matching LCM messages found for vehicle '{vehicle}'.")

    server = PlaybackServer(
        events=events,
        vehicle_name=vehicle,
        log_path=args.log,
        http_port=args.http_port,
        ws_port=args.ws_port,
        initial_speed=args.speed,
        swap_xy=args.swap_xy,
        lcm_types_path=args.lcm_types_path,
    )

    asyncio.run(server.start())


if __name__ == '__main__':
    main()
