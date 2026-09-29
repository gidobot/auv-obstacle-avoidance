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
import math
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


def parse_mission(path: str):
    """Planned waypoints from an acfr-lcm mission XML, as [[north, east], ...].

    Mission positions are NED, so x is north and y is east -- the same order
    the browser client's top-down map expects (it plots vehicle_wx against
    vehicle_y, which LiveServer fills from the gridmap's north and east).

    Only the altitude-mode primitives are returned.  The depth-mode ones are
    the launch and descent, all sitting at the same point, so drawing them
    would put a meaningless spur on the planned track.
    """
    import xml.etree.ElementTree as ET
    survey, every = [], []
    for prim in ET.parse(path).getroot().findall("primitive"):
        g = prim.find("goto")
        if g is None:
            continue
        pos, dep = g.find("position"), g.find("depth")
        if pos is None:
            continue
        pt = [float(pos.get("x", "nan")), float(pos.get("y", "nan"))]
        if pt[0] != pt[0] or pt[1] != pt[1]:
            continue
        every.append(pt)
        if dep is not None and (dep.get("mode") or "") == "altitude":
            survey.append(pt)
    return survey if len(survey) > 1 else every


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


# Log playback runs the mapper over recorded sensor data and so needs the
# pybind extension.  Live mode does not: it renders the OA_GRIDMAP the
# deployed oa-mapper node already publishes.  Importing lazily lets --live run
# anywhere LCM reaches -- notably inside the acfr_sitl container, which builds
# the C++ core for the node but not the Python extension.
try:
    from occupancy_map_cpp import (
        ObstacleMapper, OccupancyMapConfig,
        DVLConfig, SonarConfig, AltimeterConfig,
        Pose, SensorType,
        DVLMeasurement, AltimeterMeasurement, SonarMeasurement,
    )
    _HAVE_MAPPER = True
    _MAPPER_IMPORT_ERROR = None
except ImportError as exc:                       # live mode still works
    _HAVE_MAPPER = False
    _MAPPER_IMPORT_ERROR = exc
    ObstacleMapper = OccupancyMapConfig = None
    DVLConfig = SonarConfig = AltimeterConfig = None
    Pose = SensorType = None
    DVLMeasurement = AltimeterMeasurement = SonarMeasurement = None

# ---------------------------------------------------------------------------
# Browser HTML client — reuse the 3D visualizer's client unchanged
# ---------------------------------------------------------------------------
try:
    from visualizer import HTML_CLIENT_3D
except ImportError as _exc:
    # Report the real reason.  This used to claim the file was missing for any
    # ImportError, including one raised *inside* visualizer.py, which sent the
    # last person looking for a file that was sitting right there.
    _msg = f"cannot import HTML_CLIENT_3D from visualizer.py: {_exc}"
    print(f"WARNING: {_msg}", file=sys.stderr)
    HTML_CLIENT_3D = (
        "<html><body style='font:14px system-ui;padding:2rem'>"
        f"<h3>Browser client unavailable</h3><pre>{_msg}</pre></body></html>")

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
// Latest sensor returns, in world NED.  Each entry is [north, east, depth]
// (or null for a beam with no return); the plot's z axis is -depth.
let dvlHits = [], sonarHit = null, altHit = null;

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
      dvlHits  = msg.dvl_hit_xy || [];
      sonarHit = msg.sonar_hit_xy || null;
      altHit   = msg.alt_hit_xy || null;
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

// --- Sensor returns -------------------------------------------------------
// The rays show where the seafloor estimate is actually coming from, which is
// the whole point of watching this live: a surface that stops growing because
// every beam has dropped out looks identical to one the vehicle has simply
// not reached yet.
//
// Rays are drawn as a single trace with nulls separating the segments —
// Plotly breaks the line at a null, so three beams cost one trace, not three.

// A hit is [north, east, depth].  Two-element hits (no depth) cannot be
// placed in 3-D at all, so they are dropped rather than plotted at NaN.
function hasDepth(h) { return h && h.length > 2 && h[2] != null; }

function rayCoords(hits) {
  const p = trail.length ? trail[trail.length-1] : null;
  const x = [], y = [], z = [];
  if (!p) return { x, y, z };
  for (const h of hits) {
    if (!hasDepth(h)) continue;
    x.push(p[0], h[0], null);
    y.push(p[1], h[1], null);
    z.push(p[2], -h[2], null);
  }
  return { x, y, z };
}
function hitCoords(hits) {
  const valid = hits.filter(hasDepth);
  return { x: valid.map(h => h[0]), y: valid.map(h => h[1]),
           z: valid.map(h => -h[2]) };
}
function dvlRayTrace() {
  const c = rayCoords(dvlHits);
  return { type: 'scatter3d', mode: 'lines', x: c.x, y: c.y, z: c.z,
           line: { color: 'rgba(120,230,110,0.55)', width: 2 },
           hoverinfo: 'skip', showlegend: false, name: 'DVL beams' };
}
function dvlHitTrace() {
  const c = hitCoords(dvlHits);
  return { type: 'scatter3d', mode: 'markers', x: c.x, y: c.y, z: c.z,
           marker: { color: 'rgba(150,255,130,0.95)', size: 4 },
           hovertemplate: 'DVL  N %{x:.1f}  E %{y:.1f}<extra></extra>',
           showlegend: false, name: 'DVL returns' };
}
function sonarRayTrace() {
  const c = rayCoords(sonarHit ? [sonarHit] : []);
  return { type: 'scatter3d', mode: 'lines', x: c.x, y: c.y, z: c.z,
           line: { color: 'rgba(80,160,255,0.6)', width: 2 },
           hoverinfo: 'skip', showlegend: false, name: 'Sonar beam' };
}
function sonarHitTrace() {
  const c = hitCoords(sonarHit ? [sonarHit] : []);
  return { type: 'scatter3d', mode: 'markers', x: c.x, y: c.y, z: c.z,
           marker: { color: 'rgba(120,190,255,0.95)', size: 5 },
           hovertemplate: 'Sonar  N %{x:.1f}  E %{y:.1f}<extra></extra>',
           showlegend: false, name: 'Sonar return' };
}
// The altimeter gets its own colour and a bigger marker because it is the only
// beam that reads what is directly underneath.  When it and the DVL fan
// disagree, the altimeter is the one over the ground the vehicle is about to
// fly into, and that gap is the thing worth being able to see.
function altRayTrace() {
  const c = rayCoords(altHit ? [altHit] : []);
  return { type: 'scatter3d', mode: 'lines', x: c.x, y: c.y, z: c.z,
           line: { color: 'rgba(255,190,80,0.75)', width: 3 },
           hoverinfo: 'skip', showlegend: false, name: 'Altimeter beam' };
}
function altHitTrace() {
  const c = hitCoords(altHit ? [altHit] : []);
  return { type: 'scatter3d', mode: 'markers', x: c.x, y: c.y, z: c.z,
           marker: { color: 'rgba(255,200,90,1.0)', size: 6,
                     line: { color: 'rgba(180,120,20,0.9)', width: 1 } },
           hovertemplate: 'Altimeter  depth %{customdata:.2f} m<extra></extra>',
           customdata: c.z.map(v => -v),
           showlegend: false, name: 'Altimeter return' };
}

// ---------------------------------------------------------------------------
// Layout (used only once at init — never re-applied so camera is preserved)
// ---------------------------------------------------------------------------
function makeLayout(tm) {
  return {
    paper_bgcolor: '#111', plot_bgcolor: '#111',
    font:   { color: '#ccc', family: "'Menlo','Consolas',monospace" },
    margin: { l: 0, r: 80, t: 36, b: 0 },
    title:  { text: 'Seafloor Terrain — north right, east up, as the top-down view',
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
      // Open oriented like the top-down view so the two can be read against
      // each other: north to the right, east up the screen.  up = +y makes
      // east the screen vertical, and an eye that is mostly +z looks down the
      // way the 2-D map does, with just enough offset left in to show relief.
      // Both views index the same raster the same way (data[iy*nx + ix], x
      // north, y east); only the presentation differed, so this is the whole
      // of the correspondence.  Drag still rotates it freely.
      camera: { eye: { x: 0.15, y: -0.15, z: 2.0 }, up: { x: 0, y: 1, z: 0 } },
    },
  };
}

// ---------------------------------------------------------------------------
// Init (first terrain_map received)
// ---------------------------------------------------------------------------
function initPlot() {
  document.getElementById('loading').style.display = 'none';
  Plotly.newPlot('plot',
    [surfaceTrace(terrainMap), trailTrace(), vehicleTrace(),
     dvlRayTrace(), dvlHitTrace(), sonarRayTrace(), sonarHitTrace(),
     altRayTrace(), altHitTrace()],
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

  const dRay = rayCoords(dvlHits),  dHit = hitCoords(dvlHits);
  const sRay = rayCoords(sonarHit ? [sonarHit] : []);
  const sHit = hitCoords(sonarHit ? [sonarHit] : []);
  const aRay = rayCoords(altHit ? [altHit] : []);
  const aHit = hitCoords(altHit ? [altHit] : []);
  Plotly.restyle('plot', {
    x: [dRay.x, dHit.x, sRay.x, sHit.x, aRay.x, aHit.x],
    y: [dRay.y, dHit.y, sRay.y, sHit.y, aRay.y, aHit.y],
    z: [dRay.z, dHit.z, sRay.z, sHit.z, aRay.z, aHit.z],
  }, [3, 4, 5, 6, 7, 8]);
  Plotly.restyle('plot', { customdata: [aHit.z.map(v => -v)] }, [8]);
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
# Sensor geometry — world-frame beam hits and the bathymetry they accumulate
#
# Both the log-playback path and the live path need these, and live mode has
# to run without the compiled extension (the acfr_sitl container builds the
# C++ core for the oa-mapper node but not the Python bindings).  So the beam
# geometry is reproduced here in plain numpy rather than read off DVLConfig.
# ---------------------------------------------------------------------------

#: (slant_angle_deg, heading_offset_deg) per beam, mirroring the defaults in
#: oa_mapper::DVLConfig::beams.  oa-mapper exposes no bot_param key for these
#: and the LCM messages do not carry them, so there is nothing to read the
#: real geometry from -- a vehicle with a different head needs this changed.
_DVL_BEAMS = [(20.0, 0.0), (20.0, 120.0), (20.0, 240.0)]

#: Fallbacks for sensor limits that live mode cannot read from the gridmap
#: message.  They match the C++ library defaults, not any particular vehicle;
#: pass --sonar-max-range / --altimeter-max-range to match a real config.
_SONAR_MAX_RANGE_DEFAULT = 12.0
_ALT_MAX_RANGE_DEFAULT   = 100.0
_VEHICLE_LENGTH_DEFAULT  = 2.0
_SONAR_MIN_DEPTH_DEFAULT = 1.0


def dvl_beam_directions(beams=_DVL_BEAMS) -> np.ndarray:
    """Unit beam vectors in the vehicle frame, columns (forward, starboard, down).

    Same construction as ``DVLConfig::beam_directions_3d`` in the C++ library.
    """
    dirs = np.empty((len(beams), 3))
    for i, (slant_deg, h_off_deg) in enumerate(beams):
        s = math.radians(slant_deg)
        h = math.radians(h_off_deg)
        dirs[i] = (math.sin(s) * math.cos(h),   # forward
                   math.sin(s) * math.sin(h),   # starboard
                   math.cos(s))                 # down
    return dirs


def dvl_hits_world(nav_x: float, nav_y: float, nav_depth: float,
                   heading_rad: float, ranges: np.ndarray,
                   dirs: np.ndarray, hit_surface: np.ndarray) -> list:
    """Per-beam seafloor return as ``[north, east, depth]``, or None if no return.

    Depth is carried alongside the horizontal position because the same points
    feed three consumers: the top-down overlay (which reads the first two
    elements), the 3-D view, and the bathymetry accumulator.  Computing the
    geometry once keeps them from drifting apart.
    """
    dirs = np.asarray(dirs)
    n = min(len(ranges), len(dirs))   # guard against nbeams < configured beams
    cos_h = np.cos(heading_rad)
    sin_h = np.sin(heading_rad)
    hits = []
    for i in range(n):
        if not hit_surface[i] or ranges[i] <= 0:
            hits.append(None)
            continue
        r = ranges[i]
        fwd, stbd, down = dirs[i, 0], dirs[i, 1], dirs[i, 2]
        dx_fwd, dx_stbd = r * fwd, r * stbd
        hits.append([
            float(nav_x + dx_fwd * cos_h - dx_stbd * sin_h),
            float(nav_y + dx_fwd * sin_h + dx_stbd * cos_h),
            float(nav_depth + r * down),
        ])
    return hits


def altimeter_hit_world(nav_x: float, nav_y: float, nav_depth: float,
                        range_m: float, hit: bool) -> Optional[list]:
    """Altimeter return as ``[north, east, depth]``, or None.

    The dedicated vertical beam, so the return sits directly under the vehicle.
    It is the only sensor that sees what is straight below: the DVL fan at 20
    degrees lands 0.36*altitude out to each side and straddles anything
    narrower than that, which on a sawtooth is most of a tooth.
    """
    if not hit or range_m <= 0:
        return None
    return [float(nav_x), float(nav_y), float(nav_depth + range_m)]


def sonar_hit_world(nav_x: float, nav_y: float, nav_depth: float,
                    heading_rad: float, range_m: float,
                    vehicle_length: float, hit: bool) -> Optional[list]:
    """Forward-sonar return as ``[north, east, depth]``, or None.

    The beam looks horizontally forward, so the return sits at vehicle depth.
    """
    if not hit or range_m <= 0:
        return None
    total = vehicle_length / 2.0 + range_m
    return [float(nav_x + total * np.cos(heading_rad)),
            float(nav_y + total * np.sin(heading_rad)),
            float(nav_depth)]


"""The 3-D terrain map is built from sensor returns only.

There used to be a manifold_world_points() here that projected the occupancy
map's manifold into the world and fed it to the bathymetry.  It is gone on
purpose.  The manifold is a 2-D vehicle-relative construct that exists to give
the planner a defined command everywhere, so compute_manifold fills every
column whether or not anything was seen: behind the vehicle an unobserved
column takes the grid floor, ahead of it an unobserved column extends the last
observation forward and is flagged observed while doing so.  Those are the
right thing for control and inventions to a map.  Painted in, they laid 12 m of
flat seabed at a constant 15.79 m either side of a stationary vehicle, and the
accumulator keeps the shallowest value it ever sees, so the fabrication buried
the real terrain permanently.

Even the honest part of it arrives through a 2-D along-track projection that
assumes the map was built on the current heading, and through voxels that
persist across turns.  Raw returns have none of that: each one is a position in
the world at the instant it was measured.

The manifold is still published and still drawn in the profile view, which is
where a 2-D prediction belongs.
"""


class TerrainAccumulator:
    """Sparse world-frame bathymetry raster built from sensor returns.

    Each cell keeps the *shallowest* depth ever observed in it.  That is the
    surface the survey has to clear: a beam grazing a cliff face reports
    something deeper than the ridge line above it, and averaging the two would
    quietly bury the ridge.

    Shared by log playback and live mode so both render the same seafloor.
    """

    #: Target cell size (m).  The sensors sample far finer than this along
    #: track -- at 0.5 m/s and 8 Hz the DVL lands a triple every 6 cm -- so
    #: the raster, not the data, is what limits detail.  At the 2 m this used
    #: to be, the sawtooth test terrain's 70 deg face (7.2 m horizontal) was
    #: 3.6 cells wide and read as a smooth ramp.
    CELL_M = 0.5

    #: ...but the whole raster is re-serialised on every broadcast, so cap the
    #: cell count and coarsen instead of blowing up the payload on a big area.
    MAX_CELLS = 40000

    def __init__(self, ox: float, oy: float, nx: int, ny: int,
                 dx: float = CELL_M, dy: float = CELL_M,
                 mission_path: Optional[list] = None):
        self.ox, self.oy = float(ox), float(oy)
        self.nx, self.ny = int(nx), int(ny)
        self.dx, self.dy = float(dx), float(dy)
        self.mission_path = mission_path or []
        self.height_map = np.full((self.ny, self.nx), np.nan)
        self.dirty = False

    @classmethod
    def around(cls, points, margin: float = 15.0,
               cell: Optional[float] = None,
               mission_path: Optional[list] = None,
               fallback_half_extent: float = 120.0) -> 'TerrainAccumulator':
        """Grid covering ``points`` (``[north, east]`` pairs) plus a margin.

        The margin has to clear the widest the sensor footprint ever gets,
        which is the DVL's 0.36*altitude at the highest altitude flown -- about
        11 m on a descent from the surface over 30 m of water.  Much beyond
        that and the raster is mostly empty: a two-leg mission 1.5 m wide would
        sit in a 62 m-wide grid at the 30 m this used to be, and the 3-D view
        framed 40x more empty seafloor than surveyed.
        """
        cell = cls.CELL_M if cell is None else cell
        pts = [p for p in points if p is not None]
        if pts:
            ns = [p[0] for p in pts]
            es = [p[1] for p in pts]
            ox, oy = min(ns) - margin, min(es) - margin
            span_n = max(ns) + margin - ox
            span_e = max(es) + margin - oy
        else:
            ox = oy = -fallback_half_extent
            span_n = span_e = 2 * fallback_half_extent

        # Coarsen if the area would blow past the cell budget.  Scaling both
        # axes by the same factor keeps the cells square, so the top-down view
        # stays undistorted.
        n_at_cell = (span_n / cell) * (span_e / cell)
        if n_at_cell > cls.MAX_CELLS:
            cell *= math.sqrt(n_at_cell / cls.MAX_CELLS)

        nx = max(4, int(np.ceil(span_n / cell)))
        ny = max(4, int(np.ceil(span_e / cell)))
        return cls(ox, oy, nx, ny, cell, cell, mission_path)

    def clear(self) -> None:
        self.height_map[:] = np.nan
        self.dirty = False

    def record(self, world_x: float, world_y: float, depth: float) -> None:
        """Record one terrain observation, keeping the shallowest per cell."""
        if not np.isfinite(depth) or depth < 0.0:
            return
        ix = int(np.floor((world_x - self.ox) / self.dx))
        iy = int(np.floor((world_y - self.oy) / self.dy))
        if 0 <= ix < self.nx and 0 <= iy < self.ny:
            existing = self.height_map[iy, ix]
            if np.isnan(existing) or depth < existing:
                self.height_map[iy, ix] = depth
                self.dirty = True

    def build_msg(self) -> str:
        """Serialise the raster for the browser.

        Unexplored cells are encoded as JSON null so the client can render them
        in a distinct colour without dragging the depth scale around.
        """
        valid = self.height_map[~np.isnan(self.height_map)]
        if len(valid) >= 2:
            min_z, max_z = float(np.min(valid)), float(np.max(valid))
            if max_z - min_z < 1.0:
                max_z = min_z + 1.0        # prevent a degenerate colour range
        else:
            min_z, max_z = 5.0, 25.0       # defaults before enough data arrives

        return json.dumps({
            'type': 'terrain_map',
            'nx': self.nx, 'ny': self.ny,
            'dx': self.dx, 'dy': self.dy,
            'ox': self.ox, 'oy': self.oy,
            'minZ': min_z, 'maxZ': max_z,
            'data': [None if np.isnan(v) else float(v)
                     for v in self.height_map.flatten()],
            'mission_path': self.mission_path,
        })


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
        lcm_types_path: str = _DEFAULT_LCM_TYPES_PATH,
        sonar_max_range: Optional[float] = None,
        altimeter_max_range: Optional[float] = None,
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

        # DVLConfig kept for beam-geometry hit-XY visualisation
        self._dvl_cfg = DVLConfig()

        # Build mapper and store type constructors
        omap_cfg = OccupancyMapConfig()
        self._backend = 'cpp'
        self._Pose                 = Pose
        self._SensorType           = SensorType
        self._DVLMeasurement       = DVLMeasurement
        self._AltimeterMeasurement = AltimeterMeasurement
        self._SonarMeasurement     = SonarMeasurement
        # Sonar/altimeter configs kept as Python objects for threshold checks.
        # These must match the vehicle's bot_param values (e.g. cheryl.cfg
        # sonar_max_range / altimeter_max_range) or playback will classify
        # no-returns differently than the vehicle did — the defaults here are
        # the reference config, not any particular vehicle's.
        self._sonar_max_range = (sonar_max_range if sonar_max_range is not None
                                 else SonarConfig().max_range)
        self._alt_max_range   = (altimeter_max_range if altimeter_max_range is not None
                                 else AltimeterConfig().max_range)

        self.mapper = ObstacleMapper(
            omap_cfg, self._dvl_cfg, SonarConfig(), AltimeterConfig()
        )

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
        self._alt_hit_xy: Optional[list] = None
        self._elapsed_s: float = 0.0

        # LCM decoders (loaded lazily after path is set)
        self._nav_t = None
        self._alt_t = None
        self._btk_t = None
        self._isa_t = None

        # Sparse 3D height map — filled as sensors fire, broadcast periodically
        self._terrain = self._init_height_map()
        self._terrain_map_msg: str = self._terrain.build_msg()
        self._hmap_last_sent: float = 0.0

    def _load_decoders(self) -> None:
        _ensure_lcm_path(self.lcm_types_path)
        from acfrlcm import auv_acfr_nav_t
        from senlcm import nucleus_altimeter_t, nucleus_bottomtrack_t, isa500_t
        self._nav_t = auv_acfr_nav_t
        self._alt_t = nucleus_altimeter_t
        self._btk_t = nucleus_bottomtrack_t
        self._isa_t = isa500_t

    def _init_height_map(self) -> 'TerrainAccumulator':
        """Scan nav events for the track bounds, then size an empty height map."""
        _ensure_lcm_path(self.lcm_types_path)
        from acfrlcm import auv_acfr_nav_t

        track = []
        for _utime, suffix, raw in self.events:
            if suffix == 'ACFR_NAV':
                try:
                    msg = auv_acfr_nav_t.decode(raw)
                    track.append((msg.y, msg.x) if self.swap_xy else (msg.x, msg.y))
                except Exception:
                    pass
        return TerrainAccumulator.around(track)

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
            # Must match oa_mapper.cpp's gating exactly, or playback does not
            # reproduce what the vehicle did.  distance_beam_valid plus the
            # DISTANCE_SENTINEL (0.0) fully describe validity; the sensor has
            # already applied its own range limit, so nothing is clamped here.
            usable = np.isfinite(ranges) & (ranges > 0.0)
            valid &= usable
            ranges = np.where(usable, ranges, 0.0)
            pose = self._Pose(north=self._nav_x, east=self._nav_y,
                             depth=self._nav_depth, heading=self._nav_heading)
            self.mapper.update_sensor(
                self._SensorType.DVL,
                self._DVLMeasurement(ranges=ranges, hit_surface=valid),
                pose,
            )
            self._dvl_hit_xy = dvl_hits_world(
                self._nav_x, self._nav_y, self._nav_depth, self._nav_heading,
                ranges, self._dvl_cfg.beam_directions_3d, valid,
            )
            for hit in self._dvl_hit_xy:
                if hit is not None:
                    self._terrain.record(*hit)

        elif suffix == 'NUCLEUS.ALTIMETER' and self._initialized:
            msg = self._alt_t.decode(raw)
            dist = msg.altimeter_distance
            # Mirrors oa_mapper.cpp.  TODO: nucleus_altimeter_t carries
            # altimeter_quality and a documented DISTANCE_SENTINEL (0.0); this
            # max-range comparison is a stand-in for data the message already
            # provides.  Replace both sides together once a quality threshold
            # has been established from logged values.
            dist_ok = np.isfinite(dist) and dist > 0.0
            hit = dist_ok and dist < self._alt_max_range - 0.05
            pose = self._Pose(north=self._nav_x, east=self._nav_y,
                             depth=self._nav_depth, heading=self._nav_heading)
            self.mapper.update_sensor(
                self._SensorType.ALTIMETER,
                self._AltimeterMeasurement(
                    range_m=min(dist, self._alt_max_range) if dist_ok else 1.0,
                    hit=hit),
                pose,
            )
            # Rasterise altimeter hit — straight-down return at vehicle position
            self._alt_hit_xy = altimeter_hit_world(
                self._nav_x, self._nav_y, self._nav_depth, dist, hit)
            if self._alt_hit_xy is not None:
                self._terrain.record(*self._alt_hit_xy)

        elif suffix == 'ISA500_FWD' and self._initialized:
            msg = self._isa_t.decode(raw)
            dist = msg.distance
            max_r = self._sonar_max_range
            # Mirrors oa_mapper.cpp.  isa500_t carries only `distance` — no
            # validity flag and no sentinel — so max range is genuinely the
            # only way to infer a no-return here.
            dist_ok = np.isfinite(dist) and dist > 0.0
            hit = dist_ok and dist < max_r - 0.1
            pose = self._Pose(north=self._nav_x, east=self._nav_y,
                             depth=self._nav_depth, heading=self._nav_heading)
            self.mapper.update_sensor(
                self._SensorType.SONAR,
                self._SonarMeasurement(
                    range_m=min(dist, max_r) if dist_ok else max_r, hit=hit),
                pose,
            )
            self._sonar_hit_xy = sonar_hit_world(
                self._nav_x, self._nav_y, self._nav_depth, self._nav_heading,
                dist, self.mapper.omap.cfg.vehicle_length, hit,
            )
            # Rasterise sonar hit — skip shallow returns (surface reflections)
            if (self._sonar_hit_xy is not None
                    and self._nav_depth >= self.mapper.omap.cfg.sonar_min_depth_m):
                self._terrain.record(*self._sonar_hit_xy)

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
            'alt_hit_xy': self._alt_hit_xy,
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
            if self._terrain.dirty and (now - self._hmap_last_sent >= 2.0) and self.clients:
                self._terrain_map_msg = self._terrain.build_msg()
                self._terrain.dirty = False
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
                    self._alt_hit_xy = None
                    self._terrain.clear()
                    self._terrain_map_msg = self._terrain.build_msg()
                    self.mapper = ObstacleMapper(
                        OccupancyMapConfig(), self._dvl_cfg,
                        SonarConfig(), AltimeterConfig()
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

    The gridmap message carries the planner's own view — the occupancy grid,
    the manifold, the commanded-depth profile — but not the sensor returns
    behind it, so the beam footprints and the world-frame bathymetry are
    rebuilt here from the same raw channels oa-mapper consumes.  That is pure
    geometry, so it runs without the compiled extension.
    """

    def __init__(
        self,
        vehicle_name: str,
        http_port: int = 8082,
        ws_port: int = 8083,
        lcm_types_path: str = _DEFAULT_LCM_TYPES_PATH,
        mission: Optional[str] = None,
        sonar_max_range: Optional[float] = None,
        altimeter_max_range: Optional[float] = None,
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

        # Sensor limits.  Live mode has no OccupancyMapConfig to read, and the
        # gridmap message does not carry them, so these have to be told to us
        # or fall back to the library defaults.  They only affect which returns
        # count as hits, exactly as in playback.
        self._sonar_max_range = (sonar_max_range if sonar_max_range is not None
                                 else _SONAR_MAX_RANGE_DEFAULT)
        self._alt_max_range   = (altimeter_max_range if altimeter_max_range is not None
                                 else _ALT_MAX_RANGE_DEFAULT)
        self._vehicle_length  = _VEHICLE_LENGTH_DEFAULT
        self._sonar_min_depth = _SONAR_MIN_DEPTH_DEFAULT

        # Raw-sensor state, tracked from ACFR_NAV and rendered per sensor tick.
        self._dvl_dirs = dvl_beam_directions()
        self._nav_x = self._nav_y = self._nav_depth = self._nav_heading = 0.0
        self._have_nav = False
        self._dvl_hit_xy: list = []
        self._sonar_hit_xy: Optional[list] = None
        self._alt_hit_xy: Optional[list] = None

        # Bathymetry raster.  Sized around the mission when there is one; a
        # 50 m survey on a fixed 1 km plane is a dot in the middle.  Without a
        # mission the grid is deferred to the first nav fix so it lands on the
        # vehicle instead of on the origin.
        self._mission_path = parse_mission(mission) if mission else []
        if self._mission_path:
            print(f"mission: {len(self._mission_path)} waypoints from {mission}")
            self._terrain = TerrainAccumulator.around(
                self._mission_path, mission_path=self._mission_path)
        else:
            self._terrain = None

    # ------------------------------------------------------------------
    # LCM decode → browser state
    # ------------------------------------------------------------------

    def _on_nav(self, channel, data):
        msg = self._nav_t.decode(data)
        self._nav_x, self._nav_y = msg.x, msg.y     # NED: x north, y east
        self._nav_depth, self._nav_heading = msg.depth, msg.heading
        self._have_nav = True

        if self._terrain is None:
            # No mission to size the raster from — centre it on the first fix.
            self._terrain = TerrainAccumulator.around([(self._nav_x, self._nav_y)],
                                                      margin=120.0)
            self._terrain.dirty = True

        # Trail comes from nav, not from the gridmap: the gridmap is decimated
        # to ~1 Hz for bandwidth, which would draw the track as a dotted line.
        self._xy_trail.append([float(self._nav_x), float(self._nav_y)])
        if len(self._xy_trail) > 2000:
            self._xy_trail = self._xy_trail[-2000:]

    def _on_bottomtrack(self, channel, data):
        if not self._have_nav:
            return
        msg = self._btk_t.decode(data)
        # distance_beam is a fixed 3-element array; nbeams is unreliable
        # (often 0) in this ACFR driver configuration.  Gating must match
        # oa_mapper.cpp: distance_beam_valid plus the 0.0 sentinel describe
        # validity fully, and the sensor has already applied its range limit.
        n = min(len(msg.distance_beam), len(self._dvl_dirs))
        ranges = np.array(msg.distance_beam[:n], dtype=float)
        valid = np.array(msg.distance_beam_valid[:n], dtype=bool)
        usable = np.isfinite(ranges) & (ranges > 0.0)
        valid &= usable
        ranges = np.where(usable, ranges, 0.0)

        self._dvl_hit_xy = dvl_hits_world(
            self._nav_x, self._nav_y, self._nav_depth, self._nav_heading,
            ranges, self._dvl_dirs, valid,
        )
        for hit in self._dvl_hit_xy:
            if hit is not None:
                self._terrain.record(*hit)

    def _on_altimeter(self, channel, data):
        if not self._have_nav:
            return
        dist = self._alt_t.decode(data).altimeter_distance
        dist_ok = np.isfinite(dist) and dist > 0.0
        hit = dist_ok and dist < self._alt_max_range - 0.05
        self._alt_hit_xy = altimeter_hit_world(
            self._nav_x, self._nav_y, self._nav_depth, dist, hit)
        if self._alt_hit_xy is not None:
            self._terrain.record(*self._alt_hit_xy)

    def _on_sonar(self, channel, data):
        if not self._have_nav:
            return
        dist = self._isa_t.decode(data).distance
        # isa500_t carries only `distance` — no validity flag and no sentinel —
        # so max range is genuinely the only way to infer a no-return.
        dist_ok = np.isfinite(dist) and dist > 0.0
        hit = dist_ok and dist < self._sonar_max_range - 0.1
        self._sonar_hit_xy = sonar_hit_world(
            self._nav_x, self._nav_y, self._nav_depth, self._nav_heading,
            dist, self._vehicle_length, hit,
        )
        # Skip shallow returns: those are surface reflections, not seafloor.
        if (self._sonar_hit_xy is not None
                and self._nav_depth >= self._sonar_min_depth):
            self._terrain.record(*self._sonar_hit_xy)

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
            'dvl_hit_xy': self._dvl_hit_xy,
            'sonar_hit_xy': self._sonar_hit_xy,
            'alt_hit_xy': self._alt_hit_xy,
            'enable_dvl': True, 'enable_altimeter': True, 'enable_sonar': True,
        }
        self._latest_state = json.dumps(state)

    def _on_command(self, channel, data):
        self._last_cmd = self._command_t.decode(data)

    # ------------------------------------------------------------------
    # Servers
    # ------------------------------------------------------------------

    async def _broadcast_loop(self):
        last_terrain_sent = 0.0
        while True:
            now = time.monotonic()
            # Re-serialising the whole raster is not free, so it goes out on a
            # slow cadence; the seafloor does not move, only our knowledge of
            # it, and that grows a few cells per second.
            terrain_msg = None
            if (self.clients and self._terrain is not None
                    and self._terrain.dirty
                    and now - last_terrain_sent >= 2.0):
                terrain_msg = self._terrain.build_msg()
                self._terrain.dirty = False
                last_terrain_sent = now

            if self.clients and (terrain_msg or self._latest_state is not None):
                dead = set()
                for client in list(self.clients):
                    try:
                        if terrain_msg is not None:
                            await client.send(terrain_msg)
                        if self._latest_state is not None:
                            await client.send(self._latest_state)
                    except websockets.exceptions.ConnectionClosed:
                        dead.add(client)
                self.clients -= dead
            await asyncio.sleep(0.1)

    async def _ws_handler(self, websocket):
        self.clients.add(websocket)
        try:
            if self._terrain is not None:
                await websocket.send(self._terrain.build_msg())
            if self._latest_state is not None:
                await websocket.send(self._latest_state)
            async for _message in websocket:
                pass   # live mode has no playback controls; ignore client cmds
        finally:
            self.clients.discard(websocket)

    def _lcm_thread(self):
        # One undecodable message must not take the whole viewer down with it.
        #
        # lcm-python lets an exception raised inside a subscription callback
        # propagate out of handle_timeout, so without this a single bad message
        # on one channel escapes the loop and kills the thread.  Every other
        # channel then goes quiet too: the HTTP and WebSocket servers keep
        # serving, the page still loads, and the terrain simply never fills.
        # Nothing says why.
        #
        # The way to provoke it is a changed LCM type: regenerate the Python
        # bindings, leave the publisher running on the old fingerprint, and
        # every OA_COMMAND raises ValueError("Decode error").
        seen = set()
        while True:
            try:
                self._lc.handle_timeout(200)
            except Exception as exc:                      # noqa: BLE001
                key = f"{type(exc).__name__}:{exc}"[:120]
                if key not in seen:
                    seen.add(key)
                    print(f"WARNING: dropping undecodable LCM message ({key}). "
                          f"If this is a decode error, the generated types and "
                          f"the publisher disagree -- rebuild the lcmtypes and "
                          f"restart the publisher. Other channels continue.",
                          file=sys.stderr)

    async def start(self):
        import lcm
        _ensure_lcm_path(self.lcm_types_path)
        from acfrlcm import auv_oa_gridmap_t, auv_oa_command_t, auv_acfr_nav_t
        from senlcm import nucleus_altimeter_t, nucleus_bottomtrack_t, isa500_t
        self._gridmap_t = auv_oa_gridmap_t
        self._command_t = auv_oa_command_t
        self._nav_t = auv_acfr_nav_t
        self._alt_t = nucleus_altimeter_t
        self._btk_t = nucleus_bottomtrack_t
        self._isa_t = isa500_t

        v = self.vehicle_name
        self._lc = lcm.LCM()
        self._lc.subscribe(f"{v}.OA_GRIDMAP", self._on_gridmap)
        self._lc.subscribe(f"{v}.OA_COMMAND", self._on_command)
        # Same raw channels oa-mapper consumes — see the class docstring.
        self._lc.subscribe(f"{v}.ACFR_NAV", self._on_nav)
        self._lc.subscribe(f"{v}.NUCLEUS.BOTTOMTRACK", self._on_bottomtrack)
        self._lc.subscribe(f"{v}.NUCLEUS.ALTIMETER", self._on_altimeter)
        self._lc.subscribe(f"{v}.ISA500_FWD", self._on_sonar)
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
        print(f"  subscribing: {self.vehicle_name}"
              ".{OA_GRIDMAP, OA_COMMAND, ACFR_NAV, NUCLEUS.BOTTOMTRACK, "
              "NUCLEUS.ALTIMETER, ISA500_FWD}")
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
    parser.add_argument('--mission',
                        help='mission XML to draw as the planned track in the '
                             'top-down view (live mode)')
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
    parser.add_argument('--lcm-types-path', default=_DEFAULT_LCM_TYPES_PATH,
                        metavar='PATH',
                        help='Path to directory containing perls/lcmtypes package')
    parser.add_argument('--sonar-max-range', type=float, metavar='M',
                        help="Sonar max range (m) used to classify no-returns. Set this to "
                             "the vehicle's oa-mapper sonar_max_range so playback matches "
                             "what the vehicle did (cheryl.cfg: 20, seeker-sitl.cfg: 100)")
    parser.add_argument('--altimeter-max-range', type=float, metavar='M',
                        help="Altimeter max range (m) used to classify no-returns. Set this "
                             "to the vehicle's oa-mapper altimeter_max_range (cheryl.cfg and "
                             "seeker-sitl.cfg: 25)")
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
            mission=args.mission,
            sonar_max_range=args.sonar_max_range,
            altimeter_max_range=args.altimeter_max_range,
        )
        asyncio.run(live.start())
        return

    # --- Log-playback mode ---
    if not _HAVE_MAPPER:
        parser.error(
            f"log playback needs the occupancy_map_cpp extension "
            f"({_MAPPER_IMPORT_ERROR}).\n"
            f"Build it with auv-obstacle-avoidance/build.sh, or use --live, "
            f"which renders the OA_GRIDMAP published by the oa-mapper node and "
            f"needs no extension.")
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
        lcm_types_path=args.lcm_types_path,
        sonar_max_range=args.sonar_max_range,
        altimeter_max_range=args.altimeter_max_range,
    )

    asyncio.run(server.start())


if __name__ == '__main__':
    main()
