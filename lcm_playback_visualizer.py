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

# ---------------------------------------------------------------------------
# Interactive 3D terrain viewer (Plotly surface, served at /3d)
# %%WS_PORT%% is replaced with the actual WebSocket port at startup.
# ---------------------------------------------------------------------------
_HTML_3D = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AUV 3D Terrain</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#111; color:#ccc; font-family:'Menlo','Consolas',monospace; overflow:hidden; }
  #plot { width:100vw; height:100vh; }
  #hud { position:fixed; bottom:10px; left:12px; font-size:11px; color:#666;
         pointer-events:none; line-height:1.6; }
  #loading { position:fixed; top:50%; left:50%; transform:translate(-50%,-50%);
             font-size:13px; color:#555; }
</style>
</head>
<body>
<div id="loading">Loading Plotly…</div>
<div id="plot"></div>
<div id="hud"></div>
<script src="https://cdn.plot.ly/plotly-2.35.0.min.js"
        onerror="document.getElementById('loading').textContent='Plotly CDN unavailable — check internet connection.'">
</script>
<script>
const WS_URL = 'ws://localhost:%%WS_PORT%%';
let terrainMap = null, plotReady = false;
let trail = [];

// ---------------------------------------------------------------------------
// Interaction guard — all Plotly updates are deferred while the user has a
// mouse button held down.  This prevents Plotly's WebGL redraw from snapping
// the camera back mid-drag.
// ---------------------------------------------------------------------------
let interacting = false;
let pendingTerrainUpdate = false;   // true = terrain needs restyle on mouseup
let pendingVehicleUpdate = false;   // true = vehicle/trail need restyle on mouseup

window.addEventListener('mousedown',  () => { interacting = true;  });
window.addEventListener('touchstart', () => { interacting = true;  }, { passive: true });
window.addEventListener('mouseup',    () => { interacting = false; flushPending(); });
window.addEventListener('touchend',   () => { interacting = false; flushPending(); });

function flushPending() {
  if (!plotReady) return;
  if (pendingTerrainUpdate) { pendingTerrainUpdate = false; applyTerrainRestyle(); }
  if (pendingVehicleUpdate) { pendingVehicleUpdate = false; applyVehicleRestyle(); }
}

// ---------------------------------------------------------------------------
// WebSocket
// ---------------------------------------------------------------------------
function connect() {
  const ws = new WebSocket(WS_URL);
  ws.onopen  = () => setHud('Connected — waiting for terrain data…');
  ws.onclose = () => { setHud('Disconnected — retrying…'); setTimeout(connect, 2000); };
  ws.onerror = () => {};
  ws.onmessage = (e) => {
    let msg; try { msg = JSON.parse(e.data); } catch { return; }
    if (msg.type === 'terrain_map') {
      terrainMap = msg;
      if (!plotReady) { initPlot(); return; }
      if (interacting) { pendingTerrainUpdate = true; }
      else             { applyTerrainRestyle(); }
      updateHud();
    } else if (msg.vehicle_wx !== undefined && plotReady) {
      trail.push([msg.vehicle_wx, msg.vehicle_y, -msg.vehicle_z]);
      if (trail.length > 400) trail.shift();
      if (interacting) { pendingVehicleUpdate = true; }
      else             { applyVehicleRestyle(); }
    }
  };
}

// ---------------------------------------------------------------------------
// Z-grid helpers
// ---------------------------------------------------------------------------
function buildZGrid(tm) {
  const rows = [];
  for (let iy = 0; iy < tm.ny; iy++) {
    const row = [];
    for (let ix = 0; ix < tm.nx; ix++) {
      const v = tm.data[iy * tm.nx + ix];
      row.push(v === null ? null : -v);
    }
    rows.push(row);
  }
  return rows;
}

// ---------------------------------------------------------------------------
// Traces
// ---------------------------------------------------------------------------
function surfaceTrace(tm) {
  const xArr = Array.from({length: tm.nx}, (_, i) => tm.ox + (i + 0.5) * tm.dx);
  const yArr = Array.from({length: tm.ny}, (_, i) => tm.oy + (i + 0.5) * tm.dy);
  return {
    type: 'surface', x: xArr, y: yArr, z: buildZGrid(tm),
    colorscale: 'Viridis', reversescale: false,
    cmin: -tm.maxZ, cmax: -tm.minZ,
    showscale: true,
    colorbar: {
      title: { text: 'Depth (m)', font: { color: '#999', size: 11 } },
      tickvals: [-tm.maxZ, -(tm.minZ + tm.maxZ) / 2, -tm.minZ],
      ticktext: [tm.maxZ.toFixed(0) + ' m',
                 ((tm.minZ + tm.maxZ) / 2).toFixed(0) + ' m',
                 tm.minZ.toFixed(0) + ' m'],
      tickfont: { color: '#888', size: 10 },
      bgcolor: '#111', bordercolor: '#333',
      len: 0.55, x: 1.01, thickness: 14,
    },
    connectgaps: false,
    lighting:      { ambient: 0.7, diffuse: 0.6, roughness: 0.5, specular: 0.05 },
    lightposition: { x: 1, y: 0, z: 2 },
    name: 'Seafloor',
    hovertemplate: 'N %{x:.1f} m  E %{y:.1f} m<br>Depth: %{customdata:.1f} m<extra></extra>',
    customdata: buildZGrid(tm).map(r => r.map(v => v === null ? null : -v)),
  };
}
function trailTrace() {
  return {
    type: 'scatter3d', mode: 'lines',
    x: trail.map(p=>p[0]), y: trail.map(p=>p[1]), z: trail.map(p=>p[2]),
    line: { color: 'rgba(255,255,255,0.45)', width: 2 },
    hoverinfo: 'skip', showlegend: false, name: 'Trail',
  };
}
function vehicleTrace() {
  const p = trail.length ? trail[trail.length-1] : [0,0,0];
  return {
    type: 'scatter3d', mode: 'markers',
    x: [p[0]], y: [p[1]], z: [p[2]],
    marker: { color: '#F0997B', size: 7, line: { color: '#D85A30', width: 1.5 } },
    showlegend: false, name: 'AUV',
  };
}

// ---------------------------------------------------------------------------
// Layout (used only once at init — never re-applied so camera is preserved)
// ---------------------------------------------------------------------------
function makeLayout(tm) {
  return {
    paper_bgcolor: '#111', plot_bgcolor: '#111',
    font:   { color: '#ccc', family: "'Menlo','Consolas',monospace" },
    margin: { l: 0, r: 80, t: 36, b: 0 },
    title:  { text: 'Seafloor Terrain — LCM Playback',
              font: { color: '#666', size: 12 }, x: 0.46 },
    scene: {
      bgcolor: '#0b1622',
      xaxis: { title: 'North (m)', color: '#555', gridcolor: '#1d2d3d',
               zerolinecolor: '#2a3a4a', showspikes: false },
      yaxis: { title: 'East (m)',  color: '#555', gridcolor: '#1d2d3d',
               zerolinecolor: '#2a3a4a', showspikes: false },
      zaxis: {
        title: 'Depth (m)', color: '#555', gridcolor: '#1d2d3d',
        zerolinecolor: '#2a3a4a', showspikes: false,
        autorange: false, range: [-(tm.maxZ + 3), 3],
        tickvals:  [-tm.maxZ, -(tm.minZ + tm.maxZ) / 2, -tm.minZ, 0],
        ticktext:  [tm.maxZ.toFixed(0), ((tm.minZ+tm.maxZ)/2).toFixed(0),
                    tm.minZ.toFixed(0), '0 m'],
      },
      // Scale North and East proportionally to their actual extents so that
      // 1 m North == 1 m East regardless of the survey area shape.
      aspectmode: 'manual',
      aspectratio: (function() {
        const Lx = tm.nx * tm.dx;          // North extent (m)
        const Ly = tm.ny * tm.dy;          // East extent (m)
        const base = Math.max(Lx, Ly);     // normalise to larger dimension
        return { x: Lx / base, y: Ly / base, z: 0.35 };
      })(),
      camera: { eye: { x: 1.5, y: -1.5, z: 0.9 }, up: { x: 0, y: 0, z: 1 } },
    },
  };
}

// ---------------------------------------------------------------------------
// Init (first terrain_map received)
// ---------------------------------------------------------------------------
function initPlot() {
  document.getElementById('loading').style.display = 'none';
  Plotly.newPlot('plot',
    [surfaceTrace(terrainMap), trailTrace(), vehicleTrace()],
    makeLayout(terrainMap),
    { responsive: true, displaylogo: false,
      modeBarButtonsToRemove: ['resetCameraLastSave3d'] });
  plotReady = true;
  updateHud();
}

// ---------------------------------------------------------------------------
// Incremental updates — called only when NOT interacting
// ---------------------------------------------------------------------------
function applyTerrainRestyle() {
  const tm = terrainMap;
  Plotly.restyle('plot', { z: [buildZGrid(tm)], cmin: [-tm.maxZ], cmax: [-tm.minZ] }, [0]);
}
function applyVehicleRestyle() {
  Plotly.restyle('plot', {
    x: [trail.map(p=>p[0])], y: [trail.map(p=>p[1])], z: [trail.map(p=>p[2])],
  }, [1]);
  const p = trail[trail.length-1];
  Plotly.restyle('plot', { x: [[p[0]]], y: [[p[1]]], z: [[p[2]]] }, [2]);
}

function updateHud() {
  if (!terrainMap) return;
  const filled = terrainMap.data.filter(v => v !== null).length;
  const pct    = (100 * filled / terrainMap.data.length).toFixed(1);
  setHud(`Grid ${terrainMap.nx}×${terrainMap.ny}  ·  ${filled} cells (${pct}% explored)`
       + `  ·  depth ${terrainMap.minZ.toFixed(1)}–${terrainMap.maxZ.toFixed(1)} m`);
}
function setHud(txt) { document.getElementById('hud').textContent = txt; }

connect();
</script>
</body>
</html>
"""

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
# Shared browser-client HTML (patched for this tool's WS port + viewport)
# ---------------------------------------------------------------------------

def _build_client_html(ws_port: int) -> str:
    """Return the 2D/3D browser client HTML, patched for the given WS port.

    Shared by the log-playback server and the live LCM-subscribe server so the
    rendering is identical in both modes.
    """
    return (
        HTML_CLIENT_3D
        # Fix hardcoded WS port
        .replace("ws://localhost:8081",
                 f"ws://localhost:{ws_port}")
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
        # Render unexplored (null) height map cells as dark grey
        .replace(
            "      const z = data[ic];\n"
            "      const [r, g, b] = depthToRgb(z, minZ, maxZ);",
            "      const z = data[ic];\n"
            "      let r, g, b;\n"
            "      if (z == null) { r = g = b = 35; }\n"
            "      else { [r, g, b] = depthToRgb(z, minZ, maxZ); }"
        )
        # Remove Export Terrain button/size input; add 3D Map button
        .replace(
            "<button onclick=\"ws.send(JSON.stringify({cmd:'reset'}))\">Reset</button>\n"
            "    <button onclick=\"exportTerrain()\" title=\"Export current terrain as OBJ mesh"
            " + height-coloured textured material + PNG heightmap for Blender/Gazebo\">Export Terrain</button>\n"
            "    <label title=\"Side length of exported terrain (m), centred on origin\">Export size\n"
            "      <input type=\"number\" id=\"exportSize\" value=\"500\" min=\"50\" max=\"2000\""
            " step=\"50\" style=\"width:64px\">m\n"
            "    </label>",
            "<button onclick=\"ws.send(JSON.stringify({cmd:'reset'}))\">Reset</button>\n"
            "    <button onclick=\"window.open('/3d','_blank')\" title=\"Open interactive 3D terrain map\">3D Map ↗</button>",
        )
        # Hide configure gear button and panel (no terrain/trajectory to reconfigure in LCM mode)
        .replace(
            '<button id="cfgBtn" onclick="toggleCfg()" title="Configure simulation">&#9881;</button>',
            '',
        )
        .replace('<div id="cfgPanel">', '<div id="cfgPanel" style="display:none">')
        # Update page title and heading for LCM context
        .replace(
            '<title>AUV Obstacle Avoidance – 3D Simulator</title>',
            '<title>AUV Obstacle Avoidance – LCM Playback</title>',
        )
        .replace(
            'AUV Obstacle Avoidance Simulator – 3D Mode',
            'AUV Obstacle Avoidance – LCM Playback',
        )
    )


def _build_3d_html(ws_port: int) -> str:
    """Return the interactive 3D terrain page, patched for the given WS port."""
    return _HTML_3D.replace('%%WS_PORT%%', str(ws_port))


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
        use_cpp_backend: bool = False,
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

        # Python DVLConfig kept for beam-geometry hit-XY visualisation regardless of backend
        self._dvl_cfg = DVLConfig()

        # Build mapper and store backend-specific type constructors
        omap_cfg = OccupancyMapConfig()
        self._backend = 'python'
        if use_cpp_backend:
            try:
                import occupancy_map_cpp as _cpp
                cpp_cfg = _cpp.OccupancyMapConfig()
                for attr, val in vars(omap_cfg).items():
                    if hasattr(cpp_cfg, attr):
                        setattr(cpp_cfg, attr, val)
                self.mapper = _cpp.ObstacleMapper(
                    cpp_cfg, _cpp.DVLConfig(), _cpp.SonarConfig(), _cpp.AltimeterConfig()
                )
                self._Pose                 = _cpp.Pose
                self._SensorType           = _cpp.SensorType
                self._DVLMeasurement       = _cpp.DVLMeasurement
                self._AltimeterMeasurement = _cpp.AltimeterMeasurement
                self._SonarMeasurement     = _cpp.SonarMeasurement
                self._backend = 'cpp'
                print("Using C++ backend")
            except ImportError:
                use_cpp_backend = False
                print("C++ backend unavailable, falling back to Python")
        # Sonar/altimeter configs kept as Python objects for threshold checks
        self._sonar_max_range = SonarConfig().max_range
        self._alt_max_range   = AltimeterConfig().max_range

        if not use_cpp_backend:
            self.mapper = ObstacleMapper(
                omap_cfg, self._dvl_cfg, SonarConfig(), AltimeterConfig()
            )
            self._Pose                 = Pose
            self._SensorType           = SensorType
            self._DVLMeasurement       = DVLMeasurement
            self._AltimeterMeasurement = AltimeterMeasurement
            self._SonarMeasurement     = SonarMeasurement

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

        # Sparse 3D height map — filled as sensors fire, broadcast periodically
        self._init_height_map()
        self._terrain_map_msg: str = self._build_terrain_map_msg()
        self._hmap_last_sent: float = 0.0

    def _load_decoders(self) -> None:
        _ensure_lcm_path(self.lcm_types_path)
        from acfrlcm import auv_acfr_nav_t
        from senlcm import nucleus_altimeter_t, nucleus_bottomtrack_t, isa500_t
        self._nav_t = auv_acfr_nav_t
        self._alt_t = nucleus_altimeter_t
        self._btk_t = nucleus_bottomtrack_t
        self._isa_t = isa500_t

    def _init_height_map(self) -> None:
        """Scan nav events to determine map bounds, then initialise an empty height map."""
        _ensure_lcm_path(self.lcm_types_path)
        from acfrlcm import auv_acfr_nav_t

        xs, ys = [], []
        for _utime, suffix, raw in self.events:
            if suffix == 'ACFR_NAV':
                try:
                    msg = auv_acfr_nav_t.decode(raw)
                    north, east = (msg.y, msg.x) if self.swap_xy else (msg.x, msg.y)
                    xs.append(north); ys.append(east)
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

        self._hmap_ox: float = ox
        self._hmap_oy: float = oy
        self._hmap_nx: int = nx
        self._hmap_ny: int = ny
        self._hmap_dx: float = dx
        self._hmap_dy: float = dy
        self._height_map: np.ndarray = np.full((ny, nx), np.nan)
        self._hmap_dirty: bool = False

    def _build_terrain_map_msg(self) -> str:
        """Serialise the current sparse height map for the browser.

        Unexplored cells are encoded as JSON null so the browser can render
        them in a distinct 'unexplored' colour without affecting the depth
        colour scale.
        """
        hmap = self._height_map
        valid = hmap[~np.isnan(hmap)]
        if len(valid) >= 2:
            min_z = float(np.min(valid))
            max_z = float(np.max(valid))
            if max_z - min_z < 1.0:
                max_z = min_z + 1.0   # prevent degenerate colour range
        else:
            min_z, max_z = 5.0, 25.0  # defaults before enough data arrives

        data = [None if np.isnan(v) else float(v) for v in hmap.flatten()]
        return json.dumps({
            'type': 'terrain_map',
            'nx': self._hmap_nx, 'ny': self._hmap_ny,
            'dx': float(self._hmap_dx), 'dy': float(self._hmap_dy),
            'ox': float(self._hmap_ox), 'oy': float(self._hmap_oy),
            'minZ': min_z, 'maxZ': max_z,
            'data': data,
            'mission_path': [],
        })

    def _record_terrain_hit(self, world_x: float, world_y: float,
                            depth: float) -> None:
        """Record a terrain height observation; keep the shallowest (surface) depth."""
        if not np.isfinite(depth) or depth < 0.0:
            return
        ix = int(np.floor((world_x - self._hmap_ox) / self._hmap_dx))
        iy = int(np.floor((world_y - self._hmap_oy) / self._hmap_dy))
        if 0 <= ix < self._hmap_nx and 0 <= iy < self._hmap_ny:
            existing = self._height_map[iy, ix]
            if np.isnan(existing) or depth < existing:
                self._height_map[iy, ix] = depth
                self._hmap_dirty = True

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------

    def _nav_to_pose(self, msg):
        if self.swap_xy:
            north, east = msg.y, msg.x
        else:
            north, east = msg.x, msg.y
        return self._Pose(north=north, east=east,
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
            pose = self._Pose(north=self._nav_x, east=self._nav_y,
                             depth=self._nav_depth, heading=self._nav_heading)
            self.mapper.update_sensor(
                self._SensorType.DVL,
                self._DVLMeasurement(ranges=ranges, hit_surface=valid),
                pose,
            )
            self._dvl_hit_xy = _dvl_hit_xy(
                self._nav_x, self._nav_y, self._nav_heading,
                ranges, self._dvl_cfg, valid,
            )
            # Rasterise 3-D beam hit points into the height map
            dirs = self._dvl_cfg.beam_directions_3d   # shape (n, 3): fwd, stbd, down
            cos_h = np.cos(self._nav_heading)
            sin_h = np.sin(self._nav_heading)
            for i in range(min(len(ranges), len(dirs))):
                if not valid[i] or ranges[i] <= 0:
                    continue
                r = ranges[i]
                fwd, stbd, down = dirs[i, 0], dirs[i, 1], dirs[i, 2]
                self._record_terrain_hit(
                    self._nav_x + r * fwd * cos_h - r * stbd * sin_h,
                    self._nav_y + r * fwd * sin_h + r * stbd * cos_h,
                    self._nav_depth + r * down,
                )

        elif suffix == 'NUCLEUS.ALTIMETER' and self._initialized:
            msg = self._alt_t.decode(raw)
            dist = msg.altimeter_distance
            hit = dist > 0.0 and dist < self._alt_max_range - 0.05
            pose = self._Pose(north=self._nav_x, east=self._nav_y,
                             depth=self._nav_depth, heading=self._nav_heading)
            self.mapper.update_sensor(
                self._SensorType.ALTIMETER,
                self._AltimeterMeasurement(range_m=dist if dist > 0 else 1.0, hit=hit),
                pose,
            )
            # Rasterise altimeter hit — straight-down return at vehicle position
            if hit:
                self._record_terrain_hit(self._nav_x, self._nav_y,
                                         self._nav_depth + dist)

        elif suffix == 'ISA500_FWD' and self._initialized:
            msg = self._isa_t.decode(raw)
            dist = msg.distance
            max_r = self._sonar_max_range
            hit = 0.0 < dist < max_r - 0.1
            pose = self._Pose(north=self._nav_x, east=self._nav_y,
                             depth=self._nav_depth, heading=self._nav_heading)
            self.mapper.update_sensor(
                self._SensorType.SONAR,
                self._SonarMeasurement(range_m=dist if dist > 0 else max_r, hit=hit),
                pose,
            )
            self._sonar_hit_xy = _sonar_hit_xy(
                self._nav_x, self._nav_y, self._nav_heading,
                dist, self.mapper.omap.cfg.vehicle_length, hit,
            )
            # Rasterise sonar hit — skip shallow returns (surface reflections)
            if hit and self._nav_depth >= self.mapper.omap.cfg.sonar_min_depth_m:
                vl = self.mapper.omap.cfg.vehicle_length
                cos_h = np.cos(self._nav_heading)
                sin_h = np.sin(self._nav_heading)
                self._record_terrain_hit(
                    self._nav_x + (vl / 2.0 + dist) * cos_h,
                    self._nav_y + (vl / 2.0 + dist) * sin_h,
                    self._nav_depth,
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
            'backend': self._backend,
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

            # Broadcast updated height map every 2 s while data is flowing
            now = time.monotonic()
            if self._hmap_dirty and (now - self._hmap_last_sent >= 2.0) and self.clients:
                self._terrain_map_msg = self._build_terrain_map_msg()
                self._hmap_dirty = False
                self._hmap_last_sent = now
                dead = set()
                for client in list(self.clients):
                    try:
                        await client.send(self._terrain_map_msg)
                    except websockets.exceptions.ConnectionClosed:
                        dead.add(client)
                self.clients -= dead

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
                    self._height_map[:] = np.nan
                    self._hmap_dirty = False
                    self._terrain_map_msg = self._build_terrain_map_msg()
                    omap_cfg = OccupancyMapConfig()
                    if self._backend == 'cpp':
                        import occupancy_map_cpp as _cpp
                        cpp_cfg = _cpp.OccupancyMapConfig()
                        for attr, val in vars(omap_cfg).items():
                            if hasattr(cpp_cfg, attr):
                                setattr(cpp_cfg, attr, val)
                        self.mapper = _cpp.ObstacleMapper(
                            cpp_cfg, _cpp.DVLConfig(), _cpp.SonarConfig(), _cpp.AltimeterConfig()
                        )
                    else:
                        self.mapper = ObstacleMapper(
                            omap_cfg, self._dvl_cfg, SonarConfig(), AltimeterConfig()
                        )
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

        # Patched browser client (shared with the live LCM server).
        client_html = _build_client_html(self.ws_port).encode()
        html_3d = _build_3d_html(self.ws_port).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self_):
                content = html_3d if self_.path.startswith('/3d') else client_html
                self_.send_response(200)
                self_.send_header('Content-Type', 'text/html')
                self_.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self_.end_headers()
                self_.wfile.write(content)

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
# LiveServer — subscribe to the deployed oa-mapper grid snapshot over LCM
# ---------------------------------------------------------------------------

class LiveServer:
    """Render the live ``<VEHICLE>.OA_GRIDMAP`` channel in the browser client.

    Subscribes to the occupancy-grid snapshot (and OA command) published by the
    deployed ``oa-mapper`` process and serves the same browser visualizer used
    for log playback.  Works identically against a live vehicle, a live
    simulator, or an ``lcm-logplayer`` replay — it is all just LCM.
    """

    def __init__(
        self,
        vehicle_name: str,
        http_port: int = 8082,
        ws_port: int = 8083,
        lcm_types_path: str = _DEFAULT_LCM_TYPES_PATH,
    ):
        self.vehicle_name = vehicle_name
        self.http_port = http_port
        self.ws_port = ws_port
        self.lcm_types_path = lcm_types_path
        self.clients: set = set()

        self._latest_state: Optional[str] = None
        self._xy_trail: list = []
        self._last_cmd = None          # most recent auv_oa_command_t
        self._lc = None                # lcm.LCM handle

        # Flat top-down backdrop so the map view shows the vehicle + trail.
        # (The occupancy grid itself renders in the profile view from state.)
        self._terrain_map_msg = json.dumps({
            'type': 'terrain_map', 'nx': 2, 'ny': 2, 'dx': 1000.0, 'dy': 1000.0,
            'ox': -500.0, 'oy': -500.0, 'minZ': 0.0, 'maxZ': 1.0,
            'data': [0.5, 0.5, 0.5, 0.5], 'mission_path': [],
        })

    # ------------------------------------------------------------------
    # LCM decode → browser state
    # ------------------------------------------------------------------

    def _on_gridmap(self, channel, data):
        g = self._gridmap_t.decode(data)
        nx, nz, cx = g.nx, g.nz, g.cx
        manifold_z = [None if (z != z) else float(z) for z in g.manifold_z]
        cmd_depth_profile = [None if (d != d) else float(d) for d in g.cmd_depth]

        vehicle_x = g.grid_origin_x + cx * g.dx       # along-track coord (profile)
        seafloor_z = float(g.manifold_z[cx]) if 0 <= cx < nx else g.vehicle_z
        cmd_at_vehicle = cmd_depth_profile[cx] if 0 <= cx < nx else None
        altitude = (self._last_cmd.altitude
                    if (self._last_cmd is not None and self._last_cmd.altitude >= 0)
                    else (None if g.dvl_altitude < 0 else float(g.dvl_altitude)))

        self._xy_trail.append([float(g.vehicle_x), float(g.vehicle_y)])
        if len(self._xy_trail) > 500:
            self._xy_trail = self._xy_trail[-500:]

        horizon_back = cx * g.dx
        horizon_fwd = (nx - 1 - cx) * g.dx
        state = {
            'sim_mode': '3d', 'backend': 'oa-mapper',
            'vehicle_x': float(vehicle_x),
            'vehicle_wx': float(g.vehicle_x), 'vehicle_y': float(g.vehicle_y),
            'vehicle_z': float(g.vehicle_z), 'vehicle_heading': float(g.vehicle_heading),
            'terrain_z': seafloor_z,
            'altitude': altitude,
            'time': float(g.utime) / 1e6,
            'cmd_depth': cmd_at_vehicle,
            'dvl_altitude': None if g.dvl_altitude < 0 else float(g.dvl_altitude),
            'control_mode': g.control_mode,
            'terrain_label': f'LIVE: {self.vehicle_name}',
            'grid': list(g.grid),
            'nx': nx, 'nz': nz, 'dx': g.dx, 'dz': g.dz, 'cx': cx,
            'grid_origin_x': float(g.grid_origin_x),
            'manifold_grid_origin_x': float(g.manifold_grid_origin_x),
            'z_min': float(g.grid_origin_z),
            'z_max': float(g.grid_origin_z + nz * g.dz),
            'horizon_fwd': float(horizon_fwd), 'horizon_back': float(horizon_back),
            'vehicle_length': 2.0,
            'manifold_z': manifold_z,
            'cmd_depth_profile': cmd_depth_profile,
            'path_waypoints': [],
            'terrain_profile': [
                [float(vehicle_x - horizon_back), seafloor_z],
                [float(vehicle_x + horizon_fwd), seafloor_z],
            ],
            'xy_trail': self._xy_trail[-500:],
            'dvl_hit_xy': None, 'sonar_hit_xy': None,
            'enable_dvl': True, 'enable_altimeter': True, 'enable_sonar': True,
        }
        self._latest_state = json.dumps(state)

    def _on_command(self, channel, data):
        self._last_cmd = self._command_t.decode(data)

    # ------------------------------------------------------------------
    # Servers
    # ------------------------------------------------------------------

    async def _broadcast_loop(self):
        while True:
            if self.clients and self._latest_state is not None:
                dead = set()
                for client in list(self.clients):
                    try:
                        await client.send(self._latest_state)
                    except websockets.exceptions.ConnectionClosed:
                        dead.add(client)
                self.clients -= dead
            await asyncio.sleep(0.1)

    async def _ws_handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.send(self._terrain_map_msg)
            if self._latest_state is not None:
                await websocket.send(self._latest_state)
            async for _message in websocket:
                pass   # live mode has no playback controls; ignore client cmds
        finally:
            self.clients.discard(websocket)

    def _lcm_thread(self):
        while True:
            self._lc.handle_timeout(200)

    async def start(self):
        import lcm
        _ensure_lcm_path(self.lcm_types_path)
        from acfrlcm import auv_oa_gridmap_t, auv_oa_command_t
        self._gridmap_t = auv_oa_gridmap_t
        self._command_t = auv_oa_command_t

        self._lc = lcm.LCM()
        self._lc.subscribe(f"{self.vehicle_name}.OA_GRIDMAP", self._on_gridmap)
        self._lc.subscribe(f"{self.vehicle_name}.OA_COMMAND", self._on_command)
        threading.Thread(target=self._lcm_thread, daemon=True).start()

        client_html = _build_client_html(self.ws_port).encode()
        html_3d = _build_3d_html(self.ws_port).encode()

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self_):
                content = html_3d if self_.path.startswith('/3d') else client_html
                self_.send_response(200)
                self_.send_header('Content-Type', 'text/html')
                self_.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
                self_.end_headers()
                self_.wfile.write(content)

            def log_message(self_, fmt, *args):
                pass

        threading.Thread(
            target=lambda: http.server.HTTPServer(
                ('0.0.0.0', self.http_port), Handler).serve_forever(),
            daemon=True,
        ).start()

        print(f"Live oa-mapper viewer for vehicle {self.vehicle_name}")
        print(f"  subscribing: {self.vehicle_name}.OA_GRIDMAP / .OA_COMMAND")
        print(f"  HTTP:       http://localhost:{self.http_port}")
        print(f"  WebSocket:  ws://localhost:{self.ws_port}")
        print("Open the URL above; the occupancy grid renders in the profile view.")

        async with websockets.serve(self._ws_handler, '0.0.0.0', self.ws_port):
            await self._broadcast_loop()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='LCM log playback visualizer for AUV obstacle avoidance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('log', metavar='LOG_FILE', nargs='?',
                        help='Path to the LCM log file (omit when using --live)')
    parser.add_argument('--live', action='store_true',
                        help='Subscribe to the deployed oa-mapper OA_GRIDMAP channel '
                             'instead of replaying a log (requires --vehicle). Works '
                             'against a live vehicle, a live simulator, or lcm-logplayer.')
    parser.add_argument('--vehicle', metavar='NAME',
                        help='Vehicle name (e.g. DURHAM); auto-detected from log if omitted, '
                             'required with --live')
    parser.add_argument('--speed', type=float, default=1.0, metavar='X',
                        help='Initial playback speed multiplier (default: 1.0)')
    parser.add_argument('--http-port', type=int, default=8082, metavar='PORT',
                        help='HTTP port for browser client (default: 8082)')
    parser.add_argument('--ws-port', type=int, default=8083, metavar='PORT',
                        help='WebSocket port (default: 8083)')
    parser.add_argument('--swap-xy', action='store_true',
                        help='Swap nav.x/nav.y if vehicle uses east-first convention')
    parser.add_argument('--cpp', action='store_true', dest='use_cpp',
                        help='Use the C++ occupancy map backend (falls back to Python if unavailable)')
    parser.add_argument('--lcm-types-path', default=_DEFAULT_LCM_TYPES_PATH,
                        metavar='PATH',
                        help='Path to directory containing perls/lcmtypes package')
    args = parser.parse_args()

    _ensure_lcm_path(args.lcm_types_path)

    try:
        import lcm  # noqa: F401
    except ImportError:
        raise SystemExit("lcm package not found — install the LCM Python bindings")

    # --- Live mode: subscribe to the deployed oa-mapper grid snapshot ---
    if args.live:
        if not args.vehicle:
            parser.error("--live requires --vehicle NAME")
        live = LiveServer(
            vehicle_name=args.vehicle,
            http_port=args.http_port,
            ws_port=args.ws_port,
            lcm_types_path=args.lcm_types_path,
        )
        asyncio.run(live.start())
        return

    # --- Log-playback mode ---
    if not args.log:
        parser.error("a LOG_FILE is required unless --live is given")
    if not os.path.isfile(args.log):
        parser.error(f"Log file not found: {args.log}")

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
        use_cpp_backend=args.use_cpp,
        lcm_types_path=args.lcm_types_path,
    )

    asyncio.run(server.start())


if __name__ == '__main__':
    main()
