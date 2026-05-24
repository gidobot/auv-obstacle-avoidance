"""
Browser-based visualizer for the AUV obstacle avoidance simulator.

Runs a WebSocket server that streams simulation state to a browser client.
Open http://localhost:8080 after starting to see the visualization.

Usage:
    python visualizer.py
    # Then open http://localhost:8080 in a browser
"""

import asyncio
import json
import threading
import http.server
import os
import numpy as np
from pathlib import Path

try:
    import websockets
except ImportError:
    print("Installing websockets...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'websockets', '--break-system-packages'])
    import websockets

from occupancy_map import OccupancyMap, OccupancyMapConfig
from simulator import (
    Simulator3D, Trajectory3D, StraightTrajectory3D, ArcTrajectory3D,
    WaypointTrajectory3D, SegmentedTrajectory3D,
    make_terrain_3d, _TERRAIN_3D_REGISTRY,
    make_lawnmower_trajectory, _integrate_trajectory_path,
)



# ---------------------------------------------------------------------------
# 3D HTML client
# ---------------------------------------------------------------------------
HTML_CLIENT_3D = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>AUV Obstacle Avoidance – 3D Simulator</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { width: 100%; }
  body { background: #1a1a1a; color: #c8c8c0; font-family: 'Menlo', 'Consolas', monospace; }
  /* Reserve space for fixed panel (368px + border) so maps stay visible */
  #wrap { width: 100%; padding: 16px; transition: padding-right 0.22s ease; }
  #wrap.cfg-open { padding-right: calc(16px + 368px + 1px); }
  h1 { font-size: 16px; font-weight: 400; margin-bottom: 8px; color: #888; }
  .row { display: flex; gap: 12px; align-items: flex-start; }
  canvas { border-radius: 6px; background: #111; display: block; }
  #mapContainer { flex: 0 0 420px; }
  #profContainer { flex: 1 1 0; min-width: 0; }
  #mapCanvas  { width: 420px; height: 420px; }
  #profCanvas { width: 100%; height: 420px; }
  .controls { display: flex; gap: 14px; margin: 10px 0; align-items: center; font-size: 13px; flex-wrap: wrap; }
  .controls label { display: flex; align-items: center; gap: 6px; }
  .controls input[type=range] { width: 80px; }
  .stats { display: flex; gap: 16px; font-size: 12px; color: #999; margin-top: 6px; flex-wrap: wrap; }
  .legend { display: flex; gap: 12px; font-size: 11px; color: #888; margin-top: 8px; flex-wrap: wrap; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-swatch { width: 12px; height: 12px; border-radius: 2px; }
  #status { font-size: 12px; color: #666; margin-top: 4px; }
  .panel-label { font-size: 11px; color: #666; text-align: center; margin-bottom: 4px; }
  /* Settings panel */
  #cfgBtn { position:fixed; top:12px; right:14px; z-index:1000;
    background:#2a2a2a; border:1px solid #444; color:#aaa;
    width:32px; height:32px; border-radius:6px; cursor:pointer;
    font-size:18px; line-height:32px; text-align:center; }
  #cfgBtn:hover { background:#3a3a3a; color:#ddd; }
  #cfgPanel { position:fixed; top:0; right:-380px; width:368px; height:100vh;
    box-sizing:border-box;
    background:#181818; border-left:1px solid #2e2e2e;
    transition:right 0.22s ease; z-index:999; overflow-y:auto;
    padding:14px 16px; font-size:12px; }
  #cfgPanel.open { right:0; }
  #cfgPanel h2 { font-size:13px; color:#bbb; margin-bottom:12px; font-weight:500;
    border-bottom:1px solid #2e2e2e; padding-bottom:8px; }
  .cfg-sec { margin-bottom:14px; }
  .cfg-sec h3 { font-size:10px; color:#555; text-transform:uppercase;
    letter-spacing:.1em; margin-bottom:7px; }
  .cfg-row { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
  .cfg-row label { color:#888; width:140px; flex-shrink:0; }
  .cfg-row input[type=number], .cfg-row select {
    background:#111; border:1px solid #2e2e2e; color:#c8c8c0;
    border-radius:3px; padding:3px 6px; flex:1; min-width:0;
    font-size:12px; font-family:inherit; }
  .cfg-row input[type=number]:focus, .cfg-row select:focus { outline:none; border-color:#446; }
  .cfg-row input[type=checkbox] { width:15px; height:15px; accent-color:#5580bb; }
  #applyBtn { width:100%; padding:8px; margin-top:10px;
    background:#2a4068; border:1px solid #3a5080; color:#a8c4e8;
    border-radius:4px; cursor:pointer; font-size:13px; font-family:inherit; }
  #applyBtn:hover { background:#3a5080; }
</style>
</head>
<body>
<button id="cfgBtn" onclick="toggleCfg()" title="Configure simulation">&#9881;</button>
<div id="cfgPanel">
  <h2>&#9881; Configuration</h2>
  <div class="cfg-sec">
      <h3>3D Terrain</h3>
      <div class="cfg-row">
        <label>Type</label>
        <select id="cfgTerrain3d" onchange="refreshTerrain3dSec()">
          <option value="default">random reef (default)</option>
          <option value="classic">classic — seamount + ridge</option>
          <option value="flat">flat</option>
          <option value="seamount">seamount</option>
          <option value="ridge">ridge</option>
          <option value="canyon">canyon</option>
          <option value="slope">slope</option>
          <option value="sawtooth">sawtooth</option>
        </select>
      </div>
      <div id="sec-seamount" style="display:none">
        <div class="cfg-row"><label>Centre X (m)</label><input type="number" id="cfgSmCx" value="80" step="5"></div>
        <div class="cfg-row"><label>Centre Y (m)</label><input type="number" id="cfgSmCy" value="0" step="5"></div>
        <div class="cfg-row"><label>Radius (m)</label><input type="number" id="cfgSmRadius" value="30" min="5" max="300" step="5"></div>
        <div class="cfg-row"><label>Height (m)</label><input type="number" id="cfgSmHeight" value="20" min="1" max="50" step="1"></div>
      </div>
      <div id="sec-ridge" style="display:none">
        <div class="cfg-row"><label>Heading (&#176;)</label><input type="number" id="cfgRidgeHdg" value="90" min="0" max="360" step="5"></div>
        <div class="cfg-row"><label>Amplitude (m)</label><input type="number" id="cfgRidgeAmp" value="10" min="1" max="30" step="1"></div>
        <div class="cfg-row"><label>Period (m)</label><input type="number" id="cfgRidgePeriod" value="25" min="5" max="200" step="5"></div>
      </div>
      <div id="sec-canyon" style="display:none">
        <div class="cfg-row"><label>Width (m)</label><input type="number" id="cfgCanyonW" value="40" min="5" max="200" step="5"></div>
        <div class="cfg-row"><label>Extra depth (m)</label><input type="number" id="cfgCanyonD" value="8" min="1" max="30" step="1"></div>
      </div>
      <div id="sec-slope3d" style="display:none">
        <div class="cfg-row"><label>Angle (&#176;)</label><input type="number" id="cfgSlope3d" value="10" min="1" max="45" step="1"></div>
        <div class="cfg-row"><label>Slope heading (&#176;)</label><input type="number" id="cfgSlopeHdg" value="0" min="0" max="360" step="5"></div>
      </div>
      <div id="sec-sawtooth3d" style="display:none">
        <div class="cfg-row"><label>Slope angle (&#176;)</label><input type="number" id="cfgSaw3dSlope" value="45" min="1" max="89" step="5"></div>
        <div class="cfg-row"><label>Amplitude (m)</label><input type="number" id="cfgSaw3dAmp" value="10" min="1" max="30" step="1"></div>
        <div class="cfg-row"><label>Base depth (m)</label><input type="number" id="cfgSaw3dBase" value="40" min="5" max="200" step="5" title="Trough depth (m). Peak = base − amplitude. Keep ≥ amplitude+10 to avoid terrain at the surface."></div>
        <div class="cfg-row"><label>Flat bottom (m)</label><input type="number" id="cfgSaw3dFlat" value="0" min="0" max="100" step="1"></div>
        <div class="cfg-row"><label>Orientation (&#176;)</label><input type="number" id="cfgSaw3dOrient" value="0" min="0" max="360" step="5"></div>
        <div class="cfg-row"><label>Reverse</label><input type="checkbox" id="cfgSaw3dReverse"></div>
      </div>
    </div>
    <div class="cfg-sec">
      <h3>Trajectory</h3>
      <div class="cfg-row">
        <label>Type</label>
        <select id="cfgTraj" onchange="refreshTrajSec()">
          <option value="straight">straight</option>
          <option value="arc-left">arc-left</option>
          <option value="arc-right">arc-right</option>
          <option value="circle">circle</option>
          <option value="lawnmower" selected>lawnmower</option>
        </select>
      </div>
      <div class="cfg-row"><label>Initial heading (&#176;)</label><input type="number" id="cfgHdg" value="0" min="0" max="360" step="5"></div>
      <div class="cfg-row"><label>Initial depth (m)</label><input type="number" id="cfgInitDepth" value="0" min="0" step="0.5" title="Starting vehicle depth (m, positive down)."></div>
      <div id="sec-arc" style="display:none">
        <div class="cfg-row"><label>Arc radius (m)</label><input type="number" id="cfgArcRadius" value="60" min="5" max="500" step="5"></div>
      </div>
      <div id="sec-lawnmower" style="display:none">
        <div class="cfg-row"><label>Leg length (m)</label><input type="number" id="cfgLegLen" value="20" min="5" max="500" step="1"></div>
        <div class="cfg-row"><label>Spacing (m)</label><input type="number" id="cfgSpacing" value="7" min="1" max="100" step="1"></div>
        <div class="cfg-row"><label>Num legs</label><input type="number" id="cfgNLegs" value="20" min="1" max="80" step="1"></div>
        <div class="cfg-row"><label>Mission heading (&#176;)</label><input type="number" id="cfgMissionHdg" placeholder="same as initial" min="0" max="360" step="5"></div>
        <div class="cfg-row"><label>Turn rate (rad/s)</label><input type="number" id="cfgTurnRate" value="0.25" min="0" max="2" step="0.05"></div>
      </div>
    </div>
  <div class="cfg-sec">
    <h3>Obstacle Avoidance</h3>
    <div class="cfg-row"><label>Imaging alt (m)</label><input type="number" id="cfgImagingAlt" value="2" min="0.5" max="5" step="0.25"></div>
    <div class="cfg-row"><label>Cliff standoff (m)</label><input type="number" id="cfgStandoff" value="2" min="0.5" max="5" step="0.5"></div>
    <div class="cfg-row"><label>Obstacle height thresh (m)</label><input type="number" id="cfgObstThresh" value="1" min="0.1" max="3" step="0.1"></div>
    <div class="cfg-row"><label>Stale heading thresh (°)</label><input type="number" id="cfgStaleHeading" value="45" min="5" max="180" step="5" title="Voxels observed from a heading more than this many degrees away from the current heading are cleared."></div>
  </div>
  <div class="cfg-sec">
    <h3>Simulation</h3>
    <div class="cfg-row"><label>Backend</label><select id="cfgBackend"><option value="python">Python</option><option value="cpp">C++</option></select></div>
    <div class="cfg-row"><label>Time accel</label><input type="number" id="cfgTimeAccel" value="1" min="1" max="20" step="1"></div>
  </div>
  <button id="applyBtn" onclick="applyConfig()">Apply &amp; Restart</button>
</div>
<div id="wrap">
  <h1>AUV Obstacle Avoidance Simulator – 3D Mode</h1>
  <div class="row">
    <div id="mapContainer">
      <div class="panel-label">Top-down map view</div>
      <canvas id="mapCanvas" width="420" height="420"></canvas>
    </div>
    <div id="profContainer">
      <div class="panel-label">2D profile along heading</div>
      <canvas id="profCanvas" height="420"></canvas>
    </div>
  </div>
  <div class="controls">
    <button id="playBtn" onclick="togglePlay()">Play</button>
    <button onclick="ws.send(JSON.stringify({cmd:'reset'}))">Reset</button>
    <label>Time accel
      <input type="range" min="1" max="20" value="1" step="1" id="speedSlider"
             oninput="sendParam('time_accel',+this.value);document.getElementById('spdV').textContent=this.value+'x'">
      <span id="spdV">1x</span>
    </label>
    <label>Imaging alt
      <input type="range" min="1" max="5" value="2" step="0.25" id="altSlider"
             oninput="sendParam('imaging_altitude',+this.value);document.getElementById('altV').textContent=this.value+'m'">
      <span id="altV">2m</span>
    </label>
    <label>Standoff
      <input type="range" min="0.5" max="5" value="2" step="0.5" id="soSlider"
             oninput="sendParam('cliff_standoff',+this.value);document.getElementById('soV').textContent=this.value+'m'">
      <span id="soV">2m</span>
    </label>
    <label title="Voxels observed from a heading more than this many degrees away are cleared">Stale thresh
      <input type="range" min="5" max="180" value="45" step="5" id="staleSlider"
             oninput="sendParam('stale_heading_threshold_deg',+this.value);document.getElementById('staleV').textContent=this.value+'°'">
      <span id="staleV">45°</span>
    </label>
  </div>
  <div class="stats">
    <span id="stX">X: 0.0m</span>
    <span id="stY">Y: 0.0m</span>
    <span id="stZ">Z: 0.0m</span>
    <span id="stAlt">Alt: --</span>
    <span id="stDvlAlt">DVL: --</span>
    <span id="stHdg">Hdg: 0°</span>
    <span id="stT">T: 0.0s</span>
    <span id="stCmd">Cmd: --</span>
    <span id="stMode">Mode: --</span>
    <span id="stTerrain" style="color:#778">Terrain: --</span>
    <span id="stBackend" style="color:#556">Backend: python</span>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-swatch" style="background:#F0997B"></div> AUV</div>
    <div class="legend-item"><div class="legend-swatch" style="background:rgba(90,160,90,0.6)"></div> DVL</div>
    <div class="legend-item"><div class="legend-swatch" style="background:rgba(80,140,220,0.4)"></div> Sonar</div>
    <div class="legend-item"><div class="legend-swatch" style="background:rgba(255,180,60,0.7)"></div> Occupied</div>
    <div class="legend-item"><div class="legend-swatch" style="background:rgba(226,75,74,0.8)"></div> Manifold</div>
    <div class="legend-item"><div class="legend-swatch" style="background:rgba(100,220,255,0.7)"></div> Path</div>
    <div class="legend-item"><div class="legend-swatch" style="background:rgba(255,255,255,0.7)"></div> Trail</div>
    <div class="legend-item"><div class="legend-swatch" style="background:rgba(255,220,60,0.55)"></div> Mission path</div>
    <div class="legend-item"><div class="legend-swatch" style="background:rgba(160,255,120,0.95)"></div> DVL footprint</div>
    <div class="legend-item"><div class="legend-swatch" style="background:rgba(80,160,255,0.95)"></div> Sonar footprint</div>
  </div>
  <div id="status">Connecting...</div>
</div>

<script>
// ---- WebSocket ----
let ws, playing = true, latestState = null;
let terrainMap = null;  // cached terrain raster {data, nx, ny, ox, oy, dx, dy, minZ, maxZ}
let missionPath = null; // cached planned mission waypoints [[x,y], ...]

function connect() {
  ws = new WebSocket('ws://localhost:8081');
  ws.onopen = () => { document.getElementById('status').textContent = 'Connected'; };
  ws.onclose = () => {
    document.getElementById('status').textContent = 'Disconnected - retrying...';
    setTimeout(connect, 2000);
  };
  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);
    if (msg.type === 'terrain_map') {
      terrainMap = msg;
      if (msg.mission_path && msg.mission_path.length) missionPath = msg.mission_path;
      renderTerrainMap();
      return;
    }
    latestState = msg;
    draw(msg);
  };
}

function togglePlay() {
  playing = !playing;
  document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
  ws.send(JSON.stringify({ cmd: playing ? 'play' : 'pause' }));
}

function sendParam(key, val) {
  ws.send(JSON.stringify({ cmd: 'param', key, value: val }));
}

// ---- Top-down terrain map rendering ----
// Rendered once into an offscreen canvas; overlaid each frame with trail+vehicle.
const mapW = 420, mapH = 420;
let terrainImageData = null;

// Depth-to-colour: shallow (low depth) = sandy yellow, deep = dark teal
function depthToRgb(z, minZ, maxZ) {
  const t = Math.max(0, Math.min(1, (z - minZ) / (maxZ - minZ)));
  // 0=shallow (sandy), 1=deep (dark blue)
  const r = Math.round(180 - t * 150);
  const g = Math.round(160 - t * 100);
  const b = Math.round(80  + t * 120);
  return [r, g, b];
}

function renderTerrainMap() {
  if (!terrainMap) return;
  const offCanvas = document.createElement('canvas');
  offCanvas.width = mapW; offCanvas.height = mapH;
  const offCtx = offCanvas.getContext('2d');
  const img = offCtx.createImageData(mapW, mapH);
  const { data, nx, ny, ox, oy, dx, dy, minZ, maxZ } = terrainMap;

  for (let py = 0; py < mapH; py++) {
    for (let px = 0; px < mapW; px++) {
      // Map pixel -> world (x, y)
      const wx = ox + (px / mapW) * (nx * dx);
      const wy = oy + (1 - py / mapH) * (ny * dy);  // y flipped (north up)
      // Bilinear sample
      const ix = Math.floor((wx - ox) / dx);
      const iy = Math.floor((wy - oy) / dy);
      const ic = Math.max(0, Math.min(nx - 1, ix)) + Math.max(0, Math.min(ny - 1, iy)) * nx;
      const z = data[ic];
      const [r, g, b] = depthToRgb(z, minZ, maxZ);
      const pi = (py * mapW + px) * 4;
      img.data[pi]   = r;
      img.data[pi+1] = g;
      img.data[pi+2] = b;
      img.data[pi+3] = 255;
    }
  }
  offCtx.putImageData(img, 0, 0);
  terrainImageData = offCanvas;
}

function drawTopDown(s) {
  const canvas = document.getElementById('mapCanvas');
  canvas.width = mapW; canvas.height = mapH;
  const ctx = canvas.getContext('2d');

  if (!terrainMap) {
    ctx.fillStyle = '#222'; ctx.fillRect(0, 0, mapW, mapH);
    ctx.fillStyle = '#666'; ctx.font = '13px monospace'; ctx.textAlign = 'center';
    ctx.fillText('Waiting for terrain map…', mapW/2, mapH/2);
    return;
  }

  // Draw terrain background
  if (terrainImageData) ctx.drawImage(terrainImageData, 0, 0);

  const { nx, ny, ox, oy, dx, dy } = terrainMap;
  const worldW = nx * dx, worldH = ny * dy;

  // World → pixel
  function toPixel(wx, wy) {
    return [
      (wx - ox) / worldW * mapW,
      (1 - (wy - oy) / worldH) * mapH,   // north up
    ];
  }

  // Grid lines
  ctx.strokeStyle = 'rgba(255,255,255,0.06)'; ctx.lineWidth = 0.5;
  const gx0 = Math.ceil(ox / 20) * 20;
  for (let gx = gx0; gx <= ox + worldW; gx += 20) {
    const [px] = toPixel(gx, 0); ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, mapH); ctx.stroke();
  }
  const gy0 = Math.ceil(oy / 20) * 20;
  for (let gy = gy0; gy <= oy + worldH; gy += 20) {
    const [, py] = toPixel(0, gy); ctx.beginPath(); ctx.moveTo(0, py); ctx.lineTo(mapW, py); ctx.stroke();
  }

  // Mission path (planned lawnmower track) — use cached global from terrain_map msg
  const mpath = missionPath;
  if (mpath && mpath.length > 1) {
    ctx.strokeStyle = 'rgba(255,220,60,0.55)'; ctx.lineWidth = 1.2; ctx.setLineDash([6, 4]);
    ctx.beginPath();
    for (let i = 0; i < mpath.length; i++) {
      const [px, py] = toPixel(mpath[i][0], mpath[i][1]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // Waypoint dots
    ctx.fillStyle = 'rgba(255,220,60,0.5)';
    for (const pt of mpath) {
      const [px, py] = toPixel(pt[0], pt[1]);
      ctx.beginPath(); ctx.arc(px, py, 2.5, 0, 2*Math.PI); ctx.fill();
    }
  }

  // Vehicle trail
  const trail = s.xy_trail;
  if (trail && trail.length > 1) {
    ctx.strokeStyle = 'rgba(255,255,255,0.7)'; ctx.lineWidth = 1.5; ctx.beginPath();
    for (let i = 0; i < trail.length; i++) {
      const [px, py] = toPixel(trail[i][0], trail[i][1]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();
  }

  // Use world X (vehicle_wx) for the top-down map; vehicle_x is arc-local
  // and is only valid for the profile view.
  const vwx = (s.vehicle_wx !== undefined) ? s.vehicle_wx : s.vehicle_x;

  // 2D cross-section direction line (vehicle heading)
  const [vpx, vpy] = toPixel(vwx, s.vehicle_y);
  const hLen = 24;
  const heading = s.vehicle_heading || 0;
  const dxH =  Math.cos(heading) * hLen;
  const dyH = -Math.sin(heading) * hLen;   // screen y is inverted
  ctx.strokeStyle = 'rgba(100,220,255,0.7)'; ctx.lineWidth = 1.5; ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(vpx, vpy); ctx.lineTo(vpx + dxH, vpy + dyH); ctx.stroke();
  ctx.setLineDash([]);

  // DVL beam seafloor footprints
  const dvlHits = s.dvl_hit_xy;
  if (dvlHits) {
    // Beam colours match the profile view: aft, nadir, fore
    const dvlPalette = [
      'rgba(90,220,90,0.85)',    // aft beam  (–25°)
      'rgba(160,255,120,0.95)',  // nadir / altimeter (0°)
      'rgba(90,220,90,0.85)',    // fore beam (+25°)
    ];
    for (let i = 0; i < dvlHits.length; i++) {
      if (dvlHits[i] == null) continue;
      const [px, py] = toPixel(dvlHits[i][0], dvlHits[i][1]);
      // Draw a line from vehicle to hit
      ctx.strokeStyle = dvlPalette[i] || dvlPalette[0];
      ctx.lineWidth = 1; ctx.setLineDash([2, 2]);
      ctx.beginPath(); ctx.moveTo(vpx, vpy); ctx.lineTo(px, py); ctx.stroke();
      ctx.setLineDash([]);
      // Hit dot
      ctx.fillStyle = dvlPalette[i] || dvlPalette[0];
      ctx.beginPath(); ctx.arc(px, py, i === 1 ? 4 : 3, 0, 2*Math.PI); ctx.fill();
    }
  }

  // Forward sonar seafloor footprint
  const sonarHit = s.sonar_hit_xy;
  if (sonarHit) {
    const [spx, spy] = toPixel(sonarHit[0], sonarHit[1]);
    ctx.strokeStyle = 'rgba(80,160,255,0.7)';
    ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
    ctx.beginPath(); ctx.moveTo(vpx, vpy); ctx.lineTo(spx, spy); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(80,160,255,0.95)';
    ctx.beginPath(); ctx.arc(spx, spy, 4, 0, 2*Math.PI); ctx.fill();
    ctx.strokeStyle = 'rgba(160,210,255,0.9)'; ctx.lineWidth = 1;
    ctx.stroke();
  }

  // Vehicle dot
  ctx.fillStyle = '#F0997B'; ctx.beginPath();
  ctx.arc(vpx, vpy, 5, 0, 2*Math.PI); ctx.fill();
  ctx.strokeStyle = '#D85A30'; ctx.lineWidth = 1.5; ctx.stroke();

  // Axis labels
  ctx.fillStyle = '#888'; ctx.font = '10px monospace'; ctx.textAlign = 'left';
  ctx.fillText(`X: ${s.vehicle_x.toFixed(0)}m  Y: ${s.vehicle_y.toFixed(0)}m`, 6, mapH - 6);

  // Depth colour scale
  const barX = mapW - 22, barY = 10, barH = 120, barW = 12;
  const grad = ctx.createLinearGradient(0, barY, 0, barY + barH);
  const [sr, sg, sb] = depthToRgb(terrainMap.minZ, terrainMap.minZ, terrainMap.maxZ);
  const [dr, dg, db] = depthToRgb(terrainMap.maxZ, terrainMap.minZ, terrainMap.maxZ);
  grad.addColorStop(0, `rgb(${sr},${sg},${sb})`);
  grad.addColorStop(1, `rgb(${dr},${dg},${db})`);
  ctx.fillStyle = grad; ctx.fillRect(barX, barY, barW, barH);
  ctx.strokeStyle = '#555'; ctx.lineWidth = 0.5; ctx.strokeRect(barX, barY, barW, barH);
  ctx.fillStyle = '#aaa'; ctx.font = '9px monospace'; ctx.textAlign = 'left';
  ctx.fillText(terrainMap.minZ.toFixed(0)+'m', barX + barW + 2, barY + 8);
  ctx.fillText(terrainMap.maxZ.toFixed(0)+'m', barX + barW + 2, barY + barH);
}

// ---- 2D profile view (same as original, adapted for profile canvas) ----
function drawProfile(s) {
  const canvas = document.getElementById('profCanvas');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const W = rect.width, H = 420;
  canvas.width = W * dpr; canvas.height = H * dpr;
  canvas.style.height = H + 'px';
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const ox = 50, oy = 24, gw = W - 90, gh = H - 56;
  const viewW = s.horizon_fwd + s.horizon_back;
  const viewH = s.z_max - s.z_min;
  const sx = gw / viewW, sz = gh / viewH;
  const viewLeft = s.vehicle_x - s.horizon_back;
  const vehPx = s.horizon_back * sx;

  ctx.fillStyle = '#111'; ctx.fillRect(0, 0, W, H);
  ctx.fillStyle = 'rgba(15,40,65,0.6)'; ctx.fillRect(0, 0, W, H);
  ctx.save(); ctx.translate(ox, oy);

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 0.5;
  const fg = Math.ceil(viewLeft / 2) * 2;
  for (let wX = fg; wX <= viewLeft + viewW; wX += 2) {
    const px = (wX - viewLeft) * sx;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, gh); ctx.stroke();
  }
  for (let wZ = s.z_min; wZ <= s.z_max; wZ += 2) {
    const pz = (wZ - s.z_min) * sz;
    ctx.beginPath(); ctx.moveTo(0, pz); ctx.lineTo(gw, pz); ctx.stroke();
  }
  ctx.strokeStyle = 'rgba(255,255,255,0.1)'; ctx.lineWidth = 1;
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(vehPx, 0); ctx.lineTo(vehPx, gh); ctx.stroke();
  ctx.setLineDash([]);

  // Water surface line (z=0) — solid green, drawn if within view
  if (s.z_min <= 0 && s.z_max >= 0) {
    const py0 = (0 - s.z_min) * sz;
    ctx.strokeStyle = 'rgba(80,200,90,0.9)'; ctx.lineWidth = 4;
    ctx.beginPath(); ctx.moveTo(0, py0); ctx.lineTo(gw, py0); ctx.stroke();
  }

  // Terrain
  const terrain = s.terrain_profile;
  if (terrain && terrain.length > 1) {
    ctx.fillStyle = '#4a4a3a'; ctx.beginPath(); ctx.moveTo(0, gh);
    for (let i = 0; i < terrain.length; i++) {
      const px = (terrain[i][0] - viewLeft) * sx;
      const pz = (terrain[i][1] - s.z_min) * sz;
      ctx.lineTo(px, Math.min(pz, gh));
    }
    ctx.lineTo(gw, gh); ctx.closePath(); ctx.fill();
  }

  // Occupancy voxels (drawn after terrain so they appear on top)
  const grid = s.grid, nx = s.nx, nz = s.nz;
  for (let ix = 0; ix < nx; ix++) {
    const cellWX = s.grid_origin_x + ix * s.dx;
    const cellPx = (cellWX - viewLeft) * sx, cellW = s.dx * sx;
    if (cellPx + cellW < -1 || cellPx > gw + 1) continue;
    for (let iz = 0; iz < nz; iz++) {
      const p = grid[iz * nx + ix];
      if (p > 0.55) {
        const a = (p - 0.55) / 0.45;
        ctx.fillStyle = `rgba(255,160,40,${(a * 0.65).toFixed(2)})`;
        ctx.fillRect(cellPx, (s.z_min + iz * s.dz - s.z_min) * sz, cellW + 0.5, s.dz * sz + 0.5);
      }
    }
  }

  // Manifold
  const mz = s.manifold_z;
  const mOrigin = s.manifold_grid_origin_x !== undefined
                  ? s.manifold_grid_origin_x : s.grid_origin_x;
  if (mz) {
    ctx.strokeStyle = 'rgba(226,75,74,0.85)'; ctx.lineWidth = 2; ctx.beginPath();
    let prev = -1;
    for (let i = 0; i < nx; i++) {
      if (mz[i] === null) continue;
      const x  = (mOrigin + i * s.dx - viewLeft) * sx;
      const z  = (mz[i] - s.z_min) * sz;
      if (prev < 0) { ctx.moveTo(x, z); }
      else {
        const pzv = (mz[prev] - s.z_min) * sz;
        const px2 = (mOrigin + prev * s.dx - viewLeft) * sx;
        const mid = (px2 + x) / 2;
        if (z < pzv) { ctx.lineTo(mid, pzv); ctx.lineTo(mid, z); ctx.lineTo(x, z); }
        else if (z > pzv) { ctx.lineTo(mid, pzv); ctx.lineTo(mid, z); ctx.lineTo(x, z); }
        else ctx.lineTo(x, z);
      }
      prev = i;
    }
    ctx.stroke();
  }

  // Path waypoints
  const wp = s.path_waypoints;
  if (wp && wp.length > 1) {
    ctx.strokeStyle = 'rgba(100,220,255,0.75)'; ctx.lineWidth = 2; ctx.beginPath();
    for (let i = 0; i < wp.length; i++) {
      const px = (wp[i][0] - viewLeft) * sx;
      const pz = (wp[i][1] - s.z_min) * sz;
      if (i === 0) ctx.moveTo(px, pz); else ctx.lineTo(px, pz);
    }
    ctx.stroke();
  }

  // AUV
  const auvPxZ   = (s.vehicle_z - s.z_min) * sz;
  const auvPxLen = s.vehicle_length * sx;
  const auvPxH   = Math.max(6, 0.3 * sz);

  // DVL beams – Nortek Nucleus 1000 (same angles as profile view above)
  ctx.strokeStyle = 'rgba(90,160,90,0.55)'; ctx.lineWidth = 0.8;
  const _s20p = Math.sin(20 * Math.PI / 180), _c20p = Math.cos(20 * Math.PI / 180);
  const dvlBeamsP = [0, 20 * Math.PI / 180,
                     Math.atan2(_s20p * Math.cos(2 * Math.PI / 3), _c20p)]; // ≈−10.3°
  for (const ang of dvlBeamsP) {
    ctx.beginPath();
    ctx.moveTo(vehPx, auvPxZ + auvPxH / 2);
    ctx.lineTo(vehPx + Math.sin(ang)*6*sx, auvPxZ + auvPxH/2 + Math.cos(ang)*6*sz);
    ctx.stroke();
  }

  // Sonar cone
  const sonarLen  = 12 * sx;
  const nosePx    = vehPx + auvPxLen / 2;
  const sonarHalf = 3 * Math.PI / 180;
  ctx.fillStyle = 'rgba(80,140,220,0.3)'; ctx.beginPath();
  ctx.moveTo(nosePx, auvPxZ);
  ctx.lineTo(nosePx + sonarLen, auvPxZ - Math.sin(sonarHalf)*sonarLen);
  ctx.lineTo(nosePx + sonarLen, auvPxZ + Math.sin(sonarHalf)*sonarLen);
  ctx.closePath(); ctx.fill();

  // Body
  const nose = nosePx, tail = vehPx - auvPxLen/2, fin = tail - auvPxH*0.6;
  ctx.fillStyle = '#F0997B'; ctx.strokeStyle = '#D85A30'; ctx.lineWidth = 1.2;
  ctx.beginPath();
  ctx.moveTo(nose, auvPxZ);
  ctx.lineTo(tail, auvPxZ - auvPxH/2);
  ctx.lineTo(fin,  auvPxZ - auvPxH);
  ctx.lineTo(fin,  auvPxZ + auvPxH);
  ctx.lineTo(tail, auvPxZ + auvPxH/2);
  ctx.closePath(); ctx.fill(); ctx.stroke();

  ctx.restore();

  // Axis labels
  ctx.fillStyle = '#666'; ctx.font = '10px monospace'; ctx.textAlign = 'center';
  const fl = Math.ceil(viewLeft / 5) * 5;
  for (let wX = fl; wX <= viewLeft + viewW; wX += 5) {
    const px = ox + (wX - viewLeft) * sx;
    const rel = wX - s.vehicle_x;
    ctx.fillText((rel >= 0 ? '+' : '') + rel.toFixed(0) + 'm', px, oy - 6);
  }
  ctx.textAlign = 'right';
  for (let m = s.z_min; m <= s.z_max; m += 5) {
    ctx.fillText(m.toFixed(0) + 'm', ox - 4, oy + (m - s.z_min) * sz + 3);
  }
}

function draw(s) {
  drawTopDown(s);
  drawProfile(s);

  // Stats
  const dispX = (s.vehicle_wx !== undefined) ? s.vehicle_wx : s.vehicle_x;
  document.getElementById('stX').textContent   = 'X: '   + dispX.toFixed(1) + 'm';
  document.getElementById('stY').textContent   = 'Y: '   + (s.vehicle_y || 0).toFixed(1) + 'm';
  document.getElementById('stZ').textContent   = 'Z: '   + s.vehicle_z.toFixed(2) + 'm';
  document.getElementById('stAlt').textContent = 'Alt: ' + s.altitude.toFixed(2) + 'm';
  document.getElementById('stDvlAlt').textContent = 'DVL: ' + (s.dvl_altitude != null ? s.dvl_altitude.toFixed(2)+'m' : '--');
  const hdgDeg = ((s.vehicle_heading || 0) * 180 / Math.PI + 360) % 360;
  document.getElementById('stHdg').textContent = 'Hdg: ' + hdgDeg.toFixed(1) + '°';
  document.getElementById('stT').textContent   = 'T: '   + s.time.toFixed(1) + 's';
  document.getElementById('stCmd').textContent = 'Cmd: ' + s.cmd_depth.toFixed(2) + 'm';
  const modeEl = document.getElementById('stMode');
  modeEl.textContent = 'Mode: ' + (s.control_mode || 'ALT_FOLLOW');
  const modeColors = {
    ALT_FOLLOW: '#7ec8a0', OBSTACLE_CLEAR: '#f5a623', ALT_CORRECTION: '#79b8f5',
    OBSTACLE_HOLD: '#c9a227', TAIL_CLEAR: '#b86fd4',
  };
  modeEl.style.color = modeColors[s.control_mode] || '#999';
  if (s.terrain_label) document.getElementById('stTerrain').textContent = 'Terrain: ' + s.terrain_label;
  if (s.backend) document.getElementById('stBackend').textContent = 'Backend: ' + s.backend;
}

// --- Configuration panel ---
function toggleCfg() {
  const panel = document.getElementById('cfgPanel');
  const wrap = document.getElementById('wrap');
  panel.classList.toggle('open');
  /* Keep wrap padding in lockstep with panel — do not only toggle (Apply used to desync). */
  wrap.classList.toggle('cfg-open', panel.classList.contains('open'));
  wrap.addEventListener('transitionend', () => { if (latestState) draw(latestState); }, { once: true });
}
function refreshTerrain3dSec() {
  const t = document.getElementById('cfgTerrain3d').value;
  ['seamount','ridge','canyon','slope3d','sawtooth3d'].forEach(id => {
    document.getElementById('sec-' + id).style.display =
      (t === id || (id === 'slope3d' && t === 'slope') || (id === 'sawtooth3d' && t === 'sawtooth')) ? '' : 'none';
  });
}
function refreshTrajSec() {
  const t = document.getElementById('cfgTraj').value;
  document.getElementById('sec-arc').style.display =
    ['arc-left','arc-right','circle'].includes(t) ? '' : 'none';
  document.getElementById('sec-lawnmower').style.display = t === 'lawnmower' ? '' : 'none';
}
function applyConfig() {
  const mhdgEl = document.getElementById('cfgMissionHdg');
  const cfg = {
    cmd: 'configure',
    // terrain
    terrain_3d:          document.getElementById('cfgTerrain3d').value,
    seamount_cx:        +document.getElementById('cfgSmCx').value,
    seamount_cy:        +document.getElementById('cfgSmCy').value,
    seamount_radius:    +document.getElementById('cfgSmRadius').value,
    seamount_height:    +document.getElementById('cfgSmHeight').value,
    ridge_heading:      +document.getElementById('cfgRidgeHdg').value,
    ridge_amplitude:    +document.getElementById('cfgRidgeAmp').value,
    ridge_period:       +document.getElementById('cfgRidgePeriod').value,
    canyon_width:       +document.getElementById('cfgCanyonW').value,
    canyon_depth:       +document.getElementById('cfgCanyonD').value,
    slope_angle:        +document.getElementById('cfgSlope3d').value,
    slope_heading:      +document.getElementById('cfgSlopeHdg').value,
    sawtooth3d_slope:   +document.getElementById('cfgSaw3dSlope').value,
    sawtooth3d_amp:     +document.getElementById('cfgSaw3dAmp').value,
    sawtooth3d_base:    +document.getElementById('cfgSaw3dBase').value,
    sawtooth3d_flat:    +document.getElementById('cfgSaw3dFlat').value,
    sawtooth3d_orient:  +document.getElementById('cfgSaw3dOrient').value,
    sawtooth3d_reverse:  document.getElementById('cfgSaw3dReverse').checked,
    // trajectory
    trajectory:          document.getElementById('cfgTraj').value,
    heading:            +document.getElementById('cfgHdg').value,
    initial_depth:      +document.getElementById('cfgInitDepth').value,
    arc_radius:         +document.getElementById('cfgArcRadius').value,
    leg_length:         +document.getElementById('cfgLegLen').value,
    spacing:            +document.getElementById('cfgSpacing').value,
    n_legs:             +document.getElementById('cfgNLegs').value,
    mission_heading:     mhdgEl.value === '' ? null : +mhdgEl.value,
    turn_rate:          +document.getElementById('cfgTurnRate').value,
    // obstacle avoidance + sim
    imaging_altitude:    +document.getElementById('cfgImagingAlt').value,
    cliff_standoff:      +document.getElementById('cfgStandoff').value,
    obstacle_threshold:        +document.getElementById('cfgObstThresh').value,
    stale_heading_threshold_deg: +document.getElementById('cfgStaleHeading').value,
    time_accel:                +document.getElementById('cfgTimeAccel').value,
    backend:                    document.getElementById('cfgBackend').value,
  };
  ws.send(JSON.stringify(cfg));
  document.getElementById('cfgPanel').classList.remove('open');
  document.getElementById('wrap').classList.remove('cfg-open');
}

connect();
refreshTerrain3dSec();
refreshTrajSec();
window.addEventListener('resize', () => { if (latestState) draw(latestState); });
</script>
</body>
</html>
"""


class VisualizerServer3D:
    """WebSocket + HTTP server for 3D browser visualization."""

    # Default terrain map raster parameters — auto-expanded when a mission
    # path is provided (see _compute_map_extents).
    _DEFAULT_MAP_NX:  int   = 120
    _DEFAULT_MAP_NY:  int   = 100
    _DEFAULT_MAP_DX:  float = 2.0
    _DEFAULT_MAP_DY:  float = 2.0
    _DEFAULT_MAP_OX:  float = -20.0
    _DEFAULT_MAP_OY:  float = -100.0

    def __init__(
        self,
        http_port: int = 8080,
        ws_port:   int = 8081,
        dt:        float = 0.1,
        terrain_type:   str  = 'default',
        terrain_kwargs: dict | None = None,
        trajectory:     Trajectory3D | None = None,
        initial_heading_deg: float = 0.0,
        mission_path: list | None = None,
        use_cpp_backend: bool = False,
    ):
        self.use_cpp_backend     = use_cpp_backend
        self.http_port           = http_port
        self.ws_port             = ws_port
        self.dt                  = dt
        self.time_accel          = 1
        self.playing             = False
        self.clients: set        = set()
        self.trajectory          = trajectory or StraightTrajectory3D(initial_heading_deg)
        self.initial_heading_deg = initial_heading_deg
        self.mission_path        = mission_path or []
        self.terrain_type        = terrain_type
        self.terrain_kwargs      = terrain_kwargs or {}
        self._map_nx, self._map_ny, self._map_dx, self._map_dy, \
            self._map_ox, self._map_oy = self._compute_map_extents(mission_path)
        self.terrain_label       = self._build_terrain_label()
        self.sim                 = self._create_sim()
        self._terrain_map_msg    = self._build_terrain_map_msg()

    def _terrain_feature_bbox(self):
        """Return ``(x_min, y_min, x_max, y_max)`` enclosing the main terrain
        features so the map always shows the full terrain, or ``None`` if
        the terrain type has no fixed spatial extent."""
        t  = self.terrain_type
        kw = self.terrain_kwargs
        if t == 'seamount':
            cx = kw.get('cx', 80.0)
            cy = kw.get('cy', 0.0)
            r  = kw.get('radius', 30.0)
            return cx - r, cy - r, cx + r, cy + r
        if t == 'ridge':
            period = kw.get('period', 60.0)
            return 0.0, -period, period * 4, period
        if t == 'canyon':
            w = kw.get('width', 40.0)
            return 0.0, -w, 150.0, w
        if t == 'classic':
            return -5.0, -30.0, 60.0, 30.0
        if t == 'default':
            # Random reef: extents follow mission path + margins only (no bbox)
            return None
        return None

    def _compute_map_extents(self, mission_path):
        """Return (nx, ny, dx, dy, ox, oy) covering mission path + terrain
        features + margin so neither is ever clipped."""
        margin = 20.0
        dx, dy = self._DEFAULT_MAP_DX, self._DEFAULT_MAP_DY

        # Collect all bounding points
        all_x: list = [0.0]   # always include start
        all_y: list = [0.0]

        if mission_path:
            all_x.extend(p[0] for p in mission_path)
            all_y.extend(p[1] for p in mission_path)

        bbox = self._terrain_feature_bbox()
        if bbox:
            all_x.extend([bbox[0], bbox[2]])
            all_y.extend([bbox[1], bbox[3]])

        x_min = min(all_x) - margin
        x_max = max(all_x) + margin
        y_min = min(all_y) - margin
        y_max = max(all_y) + margin

        ox = x_min
        oy = y_min
        nx = max(60, int(np.ceil((x_max - x_min) / dx)))
        ny = max(60, int(np.ceil((y_max - y_min) / dy)))
        # Cap size so JSON payload stays reasonable (~200 KB)
        if nx * ny > 80_000:
            scale = np.sqrt(nx * ny / 80_000)
            dx = dx * scale
            dy = dy * scale
            nx = int(np.ceil((x_max - x_min) / dx))
            ny = int(np.ceil((y_max - y_min) / dy))
        return nx, ny, dx, dy, ox, oy

    def _create_sim(self):
        config = OccupancyMapConfig()
        terrain_fn = make_terrain_3d(self.terrain_type, **self.terrain_kwargs)
        init_depth = self.initial_depth if hasattr(self, 'initial_depth') else 0.0
        return Simulator3D(
            omap_config=config,
            terrain_fn=terrain_fn,
            trajectory=self.trajectory,
            initial_heading_deg=self.initial_heading_deg,
            initial_depth=init_depth,
            use_cpp_backend=self.use_cpp_backend,
        )

    def _build_terrain_label(self) -> str:
        if self.terrain_type == 'default':
            label = '3D/random reef'
        else:
            label = f"3D/{self.terrain_type}"
        kw = self.terrain_kwargs
        if kw.get('angle_deg'):
            label += f" {kw['angle_deg']:.0f}°"
        if kw.get('height'):
            label += f" {kw['height']:.0f}m"
        if kw.get('slope_angle_deg'):
            label += f" {kw['slope_angle_deg']:.0f}°"
        if kw.get('amplitude'):
            label += f" {kw['amplitude']:.0f}m"
        if kw.get('orientation_deg') is not None and self.terrain_type == 'sawtooth':
            label += f" orient={kw['orientation_deg']:.0f}°"
        return label

    def _build_terrain_map_msg(self) -> str:
        """Pre-compute terrain raster and return as JSON string (sent once)."""
        sim = self.sim
        nx, ny = self._map_nx, self._map_ny
        dx, dy = self._map_dx, self._map_dy
        ox, oy = self._map_ox, self._map_oy

        data: list = []
        z_min =  1e9
        z_max = -1e9
        for iy in range(ny):
            for ix in range(nx):
                wx = ox + (ix + 0.5) * dx
                wy = oy + (iy + 0.5) * dy
                z  = sim._terrain_at(wx, wy)
                data.append(float(z))
                if z < z_min: z_min = z
                if z > z_max: z_max = z

        return json.dumps({
            'type': 'terrain_map',
            'nx': nx, 'ny': ny,
            'dx': float(dx), 'dy': float(dy),
            'ox': float(ox), 'oy': float(oy),
            'minZ': float(z_min), 'maxZ': float(z_max),
            'data': data,
            'mission_path': self.mission_path,
        })

    def _build_state_msg(self) -> str:
        sim  = self.sim   # Simulator3D
        omap = sim.omap
        cfg  = omap.cfg
        snap = omap.get_grid_snapshot()

        # Terrain profile along current heading (for profile view)
        view_left  = sim._arc_local - cfg.horizon_back
        view_right = sim._arc_local + cfg.horizon_fwd
        n_terrain  = 200
        arc_samples = np.linspace(view_left, view_right, n_terrain)
        cos_h = np.cos(sim.vehicle_heading)
        sin_h = np.sin(sim.vehicle_heading)
        terrain_profile = []
        for s in arc_samples:
            wx = sim.vehicle_x + (s - sim._arc_local) * cos_h
            wy = sim.vehicle_y + (s - sim._arc_local) * sin_h
            world_x_view = sim._arc_local + (s - sim._arc_local)
            terrain_profile.append([float(world_x_view), float(sim._terrain_at(wx, wy))])

        manifold_z = [None if np.isnan(z) else float(z)
                      for z in snap['manifold_z']]
        grid_flat  = snap['grid'].flatten().tolist()

        # vehicle_x  = arc_local (along-track grid coord — used by profile view)
        # vehicle_wx = actual world X             — used by top-down map view
        vehicle_x_prof = float(sim._arc_local)

        state = {
            'sim_mode':     '3d',
            'backend':      sim._backend,
            'vehicle_x':    vehicle_x_prof,
            'vehicle_wx':   float(sim.vehicle_x),
            'vehicle_y':    float(sim.vehicle_y),
            'vehicle_z':    float(sim.vehicle_z),
            'vehicle_heading': float(sim.vehicle_heading),
            'terrain_z':    float(sim._terrain_at(sim.vehicle_x, sim.vehicle_y)),
            'altitude':     float(sim._terrain_at(sim.vehicle_x, sim.vehicle_y) - sim.vehicle_z),
            'time':         float(sim.time),
            'cmd_depth':    float(omap.get_commanded_depth_at_vehicle()),
            'dvl_altitude': snap['dvl_altitude'],
            'control_mode': snap['control_mode'],
            'terrain_label': self.terrain_label,
            'grid':         grid_flat,
            'nx': snap['nx'], 'nz': snap['nz'],
            'dx': snap['dx'], 'dz': snap['dz'],
            'cx': snap['cx'],
            'grid_origin_x':          float(snap['grid_origin_x']),
            'manifold_grid_origin_x': float(snap['manifold_grid_origin_x']),
            'z_min': float(snap['z_min']), 'z_max': float(snap['z_max']),
            'horizon_fwd':   float(cfg.horizon_fwd),
            'horizon_back':  float(cfg.horizon_back),
            'vehicle_length': float(cfg.vehicle_length),
            'manifold_z':    manifold_z,
            'cmd_depth_profile': [float(d) if not np.isnan(d) else None
                                  for d in snap['cmd_depth']],
            'path_waypoints':  snap['path_waypoints'],
            'terrain_profile': terrain_profile,
            'xy_trail': sim.xy_trail[-500:],
            'dvl_hit_xy':   [list(p) if p is not None else None
                             for p in sim.dvl_hit_xy],
            'sonar_hit_xy': list(sim.sonar_hit_xy) if sim.sonar_hit_xy else None,
        }
        return json.dumps(state)

    def _apply_config(self, data: dict) -> None:
        """Reconfigure terrain, trajectory, and OccupancyMap params."""
        terrain_type = data.get('terrain_3d', 'default')
        terrain_kwargs: dict = {}
        if terrain_type == 'seamount':
            terrain_kwargs = dict(
                cx=float(data.get('seamount_cx', 80.0)),
                cy=float(data.get('seamount_cy', 0.0)),
                radius=float(data.get('seamount_radius', 30.0)),
                height=float(data.get('seamount_height', 20.0)),
            )
        elif terrain_type == 'ridge':
            terrain_kwargs = dict(
                ridge_heading_deg=float(data.get('ridge_heading', 90.0)),
                amplitude=float(data.get('ridge_amplitude', 10.0)),
                period=float(data.get('ridge_period', 25.0)),
            )
        elif terrain_type == 'canyon':
            terrain_kwargs = dict(
                width=float(data.get('canyon_width', 40.0)),
                extra_depth=float(data.get('canyon_depth', 8.0)),
            )
        elif terrain_type == 'slope':
            # Place the slope origin 10 m below the vehicle's initial depth
            # so the vehicle always starts at imaging altitude over the
            # slope's near end regardless of the configured initial depth.
            init_depth_val = float(data.get('initial_depth', 0.0))
            terrain_kwargs = dict(
                angle_deg=float(data.get('slope_angle', 10.0)),
                slope_heading_deg=float(data.get('slope_heading', 0.0)),
                base_depth=init_depth_val + 10.0,
            )
        elif terrain_type == 'sawtooth':
            amp = float(data.get('sawtooth3d_amp', 10.0))
            # Default base_depth ensures the peak (base_depth - amplitude)
            # stays well below the surface. Guard against user entering a
            # value that would put the peak at or above depth 0.
            base_depth = float(data.get('sawtooth3d_base', 40.0))
            base_depth = max(base_depth, amp + 5.0)
            terrain_kwargs = dict(
                slope_angle_deg=float(data.get('sawtooth3d_slope', 45.0)),
                amplitude=amp,
                base_depth=base_depth,
                flat_bottom=float(data.get('sawtooth3d_flat', 0.0)),
                orientation_deg=float(data.get('sawtooth3d_orient', 0.0)),
                reverse=bool(data.get('sawtooth3d_reverse', False)),
            )
        self.terrain_type   = terrain_type
        self.terrain_kwargs = terrain_kwargs
        self.terrain_label  = self._build_terrain_label()

        traj_type   = data.get('trajectory', 'straight')
        heading_deg = float(data.get('heading', 0.0))
        arc_radius  = float(data.get('arc_radius', 60.0))
        mhdg_raw    = data.get('mission_heading')
        mission_hdg = heading_deg if (mhdg_raw is None or mhdg_raw == '') \
                      else float(mhdg_raw)
        mission_path: list = []
        if traj_type == 'arc-left':
            traj = ArcTrajectory3D(heading_deg, radius=arc_radius, direction='left')
        elif traj_type == 'arc-right':
            traj = ArcTrajectory3D(heading_deg, radius=arc_radius, direction='right')
        elif traj_type == 'circle':
            traj = ArcTrajectory3D(heading_deg, radius=arc_radius, direction='left')
        elif traj_type == 'lawnmower':
            traj, mission_path = make_lawnmower_trajectory(
                leg_length=float(data.get('leg_length', 20.0)),
                spacing=float(data.get('spacing', 7.0)),
                n_legs=int(data.get('n_legs', 20)),
                orientation_deg=mission_hdg,
                turn_rate=float(data.get('turn_rate', 0.25)),
                survey_speed=0.5,
            )
        else:
            traj = StraightTrajectory3D(heading_deg)

        self.trajectory          = traj
        self.initial_heading_deg = heading_deg
        self.initial_depth       = float(data.get('initial_depth', 0.0))
        self.mission_path        = mission_path
        self._map_nx, self._map_ny, self._map_dx, self._map_dy, \
            self._map_ox, self._map_oy = self._compute_map_extents(mission_path)
        self.sim = self._create_sim()

        if 'backend' in data:
            self.use_cpp_backend = (data['backend'] == 'cpp')
        for key in ('imaging_altitude', 'cliff_standoff', 'obstacle_threshold',
                    'stale_heading_threshold_deg'):
            if key in data:
                setattr(self.sim.omap.cfg, key, float(data[key]))
        if 'time_accel' in data:
            self.time_accel = int(data['time_accel'])

    async def sim_loop(self):
        """Main simulation loop — steps sim and broadcasts state."""
        while True:
            if self.playing and len(self.clients) > 0:
                for _ in range(self.time_accel):
                    self.sim.step(self.dt)
                msg = self._build_state_msg()
                dead = set()
                for client in self.clients:
                    try:
                        await client.send(msg)
                    except websockets.exceptions.ConnectionClosed:
                        dead.add(client)
                self.clients -= dead
            await asyncio.sleep(self.dt)

    async def ws_handler(self, websocket):
        """Handle WebSocket client: send terrain map on connect then stream state."""
        self.clients.add(websocket)
        try:
            # Send terrain map once at connect
            await websocket.send(self._terrain_map_msg)
            async for message in websocket:
                data = json.loads(message)
                cmd  = data.get('cmd')
                if cmd == 'play':
                    self.playing = True
                elif cmd == 'pause':
                    self.playing = False
                elif cmd == 'reset':
                    self.sim = self._create_sim()
                    self._terrain_map_msg = self._build_terrain_map_msg()
                    await websocket.send(self._terrain_map_msg)
                elif cmd == 'configure':
                    self._apply_config(data)
                    self._terrain_map_msg = self._build_terrain_map_msg()
                    # Broadcast new terrain map to every connected client
                    dead = set()
                    for client in self.clients:
                        try:
                            await client.send(self._terrain_map_msg)
                        except Exception:
                            dead.add(client)
                    self.clients -= dead
                elif cmd == 'param':
                    key = data['key']
                    val = data['value']
                    if key == 'time_accel':
                        self.time_accel = int(val)
                    elif hasattr(self.sim.omap.cfg, key):
                        setattr(self.sim.omap.cfg, key, float(val))
        finally:
            self.clients.discard(websocket)

    async def start(self):
        """Start HTTP (serving 3D HTML) and WebSocket servers."""
        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self_):
                self_.send_response(200)
                self_.send_header('Content-Type', 'text/html')
                self_.end_headers()
                self_.wfile.write(HTML_CLIENT_3D.encode())

            def log_message(self_, format, *args):
                pass

        http_thread = threading.Thread(
            target=lambda: http.server.HTTPServer(
                ('0.0.0.0', self.http_port), Handler
            ).serve_forever(),
            daemon=True,
        )
        http_thread.start()
        print(f"HTTP server: http://localhost:{self.http_port}")
        print(f"WebSocket server: ws://localhost:{self.ws_port}")

        async with websockets.serve(self.ws_handler, '0.0.0.0', self.ws_port):
            await self.sim_loop()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='AUV Obstacle Avoidance 3D Visualizer',
    )

    # --- 3D terrain ---
    parser.add_argument(
        '--terrain-3d', default='default', dest='terrain_3d',
        metavar='TYPE',
        help=(
            'Terrain type, one of: '
            + ', '.join(sorted(_TERRAIN_3D_REGISTRY))
            + '  (default: %(default)s)'
        ),
    )
    parser.add_argument('--slope-angle', type=float, default=45.0, metavar='DEG',
        help='Sawtooth/slope angle in degrees (default: %(default)s)')
    parser.add_argument('--amplitude', type=float, default=10.0, metavar='M',
        help='Sawtooth tooth height in metres (default: %(default)s)')
    parser.add_argument('--flat-bottom', type=float, default=0.0, metavar='M',
        help='Flat seafloor distance between sawtooth teeth in metres (default: %(default)s)')
    parser.add_argument('--reverse', action='store_true', default=False,
        help='Reverse sawtooth direction')
    parser.add_argument('--seamount-cx', type=float, default=80.0, metavar='M')
    parser.add_argument('--seamount-cy', type=float, default=0.0, metavar='M')
    parser.add_argument('--seamount-radius', type=float, default=30.0, metavar='M')
    parser.add_argument('--seamount-height', type=float, default=14.0, metavar='M')
    parser.add_argument('--ridge-heading', type=float, default=90.0, metavar='DEG')
    parser.add_argument('--ridge-amplitude', type=float, default=8.0, metavar='M')
    parser.add_argument('--ridge-period', type=float, default=60.0, metavar='M')
    parser.add_argument('--canyon-width', type=float, default=40.0, metavar='M')
    parser.add_argument('--canyon-depth', type=float, default=8.0, metavar='M')
    parser.add_argument('--slope-heading', type=float, default=0.0, metavar='DEG')

    # --- 3D trajectory ---
    parser.add_argument(
        '--trajectory', default='lawnmower',
        choices=['straight', 'arc-left', 'arc-right', 'lawnmower', 'circle'],
        help='XY trajectory type (default: %(default)s)',
    )
    parser.add_argument('--heading', type=float, default=0.0, metavar='DEG',
        help='Initial heading in degrees (default: %(default)s)')
    parser.add_argument('--arc-radius', type=float, default=60.0, metavar='M',
        help='Arc radius in metres for arc/circle trajectories (default: %(default)s)')
    parser.add_argument('--leg-length', type=float, default=20.0, metavar='M',
        help='Lawnmower leg length in metres (default: %(default)s)')
    parser.add_argument('--spacing', type=float, default=7.0, metavar='M',
        help='Lawnmower cross-track spacing in metres (default: %(default)s)')
    parser.add_argument('--n-legs', type=int, default=20, metavar='N',
        help='Number of lawnmower legs (default: %(default)s)')
    parser.add_argument('--mission-heading', type=float, default=None, metavar='DEG',
        help='Lawnmower leg orientation; defaults to --heading')
    parser.add_argument('--turn-rate', type=float, default=0.25, metavar='RAD/S',
        help='Lawnmower corner turn rate in rad/s (default: %(default)s)')

    # --- Common ---
    parser.add_argument('--http-port', type=int, default=8080, metavar='PORT',
        help='HTTP port for browser client (default: %(default)s)')
    parser.add_argument('--ws-port', type=int, default=8081, metavar='PORT',
        help='WebSocket port (default: %(default)s)')

    args = parser.parse_args()

    # --- Terrain kwargs ---
    terrain_kwargs: dict = {}
    t3 = args.terrain_3d
    if t3 not in _TERRAIN_3D_REGISTRY:
        parser.error(
            f"Unknown terrain '{t3}'. Choose from: "
            + ', '.join(sorted(_TERRAIN_3D_REGISTRY))
        )
    if t3 == 'seamount':
        terrain_kwargs = dict(
            cx=args.seamount_cx, cy=args.seamount_cy,
            radius=args.seamount_radius, height=args.seamount_height,
        )
    elif t3 == 'ridge':
        terrain_kwargs = dict(
            ridge_heading_deg=args.ridge_heading,
            amplitude=args.ridge_amplitude,
            period=args.ridge_period,
        )
    elif t3 == 'canyon':
        terrain_kwargs = dict(width=args.canyon_width, extra_depth=args.canyon_depth)
    elif t3 == 'slope':
        terrain_kwargs = dict(
            angle_deg=args.slope_angle,
            slope_heading_deg=args.slope_heading,
        )
    elif t3 == 'sawtooth':
        terrain_kwargs = dict(
            slope_angle_deg=args.slope_angle,
            amplitude=args.amplitude,
            flat_bottom=args.flat_bottom,
            reverse=args.reverse,
        )

    # --- Trajectory ---
    mission_path: list = []
    mission_hdg = args.mission_heading if args.mission_heading is not None else args.heading
    if args.trajectory == 'arc-left':
        traj = ArcTrajectory3D(args.heading, radius=args.arc_radius, direction='left')
    elif args.trajectory == 'arc-right':
        traj = ArcTrajectory3D(args.heading, radius=args.arc_radius, direction='right')
    elif args.trajectory == 'circle':
        traj = ArcTrajectory3D(args.heading, radius=args.arc_radius, direction='left')
    elif args.trajectory == 'lawnmower':
        traj, mission_path = make_lawnmower_trajectory(
            leg_length=args.leg_length,
            spacing=args.spacing,
            n_legs=args.n_legs,
            orientation_deg=mission_hdg,
            turn_rate=args.turn_rate,
            survey_speed=0.5,
        )
    else:
        traj = StraightTrajectory3D(args.heading)

    server = VisualizerServer3D(
        http_port=args.http_port,
        ws_port=args.ws_port,
        terrain_type=args.terrain_3d,
        terrain_kwargs=terrain_kwargs,
        trajectory=traj,
        initial_heading_deg=args.heading,
        mission_path=mission_path,
    )
    traj_desc = args.trajectory
    if args.trajectory == 'lawnmower':
        turn_desc = (f'  turn={args.turn_rate:.2f}rad/s'
                     if args.turn_rate > 0 else '  square-corners')
        traj_desc += (f'  legs={args.n_legs}×{args.leg_length:.0f}m'
                      f'  spacing={args.spacing:.0f}m  hdg={mission_hdg:.0f}°' + turn_desc)
    print(f'Terrain: {server.terrain_label}  Trajectory: {traj_desc}')
    asyncio.run(server.start())


if __name__ == '__main__':
    main()
