#!/usr/bin/env python3
"""Terrain sources for the ideal-trajectory analysis.

The analysis stack is already parameterised on a terrain callable:
``analytic_ideal``, ``OracleMapper`` and ``Simulator3D`` all take
``terrain_fn(x) -> seabed depth, positive down``.  What was missing was a way
to get that callable from a real survey rather than an analytic profile.

This module supplies it for any mesh trimesh can read -- COLLADA .dae, .obj,
.ply, .stl -- and for plain grids.  A mesh is resampled once onto a regular
grid in the nav frame and sampled bilinearly after that, which keeps the cost
independent of mesh density: a dense reconstruction is paid for at load, not
per query.

Frames.  Nav is x = north, y = east, depth positive down.  Meshes are usually
Z-up in some survey frame, so the mapping is given explicitly rather than
guessed -- the bundled terrain/sawtooth.obj, for instance, has its own header
saying "X = relief axis (nav east), Y = along-crest (nav north), Z = Up",
which is x/y transposed relative to nav.

Limitation worth stating where results are quoted: the field is 2.5-D.  A
single depth per (north, east) cannot represent an overhang, and a
reconstruction containing one will be flattened to whichever surface the
interpolator lands on.  For seabed survey that is almost always what is
wanted, but it is an assumption, not a fact about the data.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable, Optional, Tuple

import numpy as np

__all__ = ["TerrainField", "from_mesh", "from_grid"]


class TerrainField:
    """A 2.5-D seabed, sampled in the nav frame.

    ``grid[i_north, i_east]`` holds depth in metres, positive down; NaN marks
    cells the survey does not cover.
    """

    def __init__(self, grid: np.ndarray, north0: float, east0: float,
                 cell: float, name: str = ""):
        self.grid = np.asarray(grid, dtype=float)
        self.north0 = float(north0)
        self.east0 = float(east0)
        self.cell = float(cell)
        self.name = name

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        n, e = self.grid.shape
        return (self.north0, self.north0 + n * self.cell,
                self.east0, self.east0 + e * self.cell)

    def depth_at(self, north, east):
        """Bilinear depth at nav (north, east).  NaN outside the survey.

        Accepts scalars or arrays and returns the same; a query that lands on
        a hole or off the edge comes back NaN rather than being clamped to the
        nearest cell, so missing survey stays visibly missing.
        """
        scalar = np.isscalar(north) and np.isscalar(east)
        north = np.atleast_1d(np.asarray(north, dtype=float))
        east = np.atleast_1d(np.asarray(east, dtype=float))
        fi = (north - self.north0) / self.cell
        fj = (east - self.east0) / self.cell
        n, m = self.grid.shape
        i0 = np.floor(fi).astype(int)
        j0 = np.floor(fj).astype(int)
        ok = (i0 >= 0) & (i0 < n - 1) & (j0 >= 0) & (j0 < m - 1)
        out = np.full(fi.shape, np.nan, dtype=float)
        if np.any(ok):
            ii, jj = i0[ok], j0[ok]
            wi, wj = (fi[ok] - ii), (fj[ok] - jj)
            out[ok] = (self.grid[ii, jj]         * (1 - wi) * (1 - wj)
                       + self.grid[ii, jj + 1]   * (1 - wi) * wj
                       + self.grid[ii + 1, jj]   * wi       * (1 - wj)
                       + self.grid[ii + 1, jj+1] * wi       * wj)
        return float(out[0]) if scalar else out

    def along(self, p0, p1) -> Callable[[float], float]:
        """terrain_fn(s) along the straight leg p0 -> p1, s from p0 in metres.

        p0/p1 are nav (north, east).  Signature matches what analytic_ideal
        and Simulator3D expect.
        """
        n0, e0 = float(p0[0]), float(p0[1])
        n1, e1 = float(p1[0]), float(p1[1])
        L = math.hypot(n1 - n0, e1 - e0)
        if L < 1e-9:
            raise ValueError("leg has zero length")
        un, ue = (n1 - n0) / L, (e1 - e0) / L

        def terrain_fn(s: float) -> float:
            return float(self.depth_at(n0 + un * s, e0 + ue * s))

        terrain_fn.length = L          # type: ignore[attr-defined]
        return terrain_fn

    def coverage(self) -> float:
        return float(np.mean(np.isfinite(self.grid)))

    def __repr__(self):
        n0, n1, e0, e1 = self.extent
        finite = self.grid[np.isfinite(self.grid)]
        d = (f"depth {finite.min():.2f}..{finite.max():.2f} m"
             if finite.size else "empty")
        return (f"TerrainField({self.name!r} {self.grid.shape[0]}x"
                f"{self.grid.shape[1]} @ {self.cell} m, north {n0:.1f}..{n1:.1f}, "
                f"east {e0:.1f}..{e1:.1f}, {d}, {100*self.coverage():.0f}% covered)")


def from_grid(depth_grid, north0, east0, cell, name="grid") -> TerrainField:
    return TerrainField(depth_grid, north0, east0, cell, name)


def from_mesh(path: str,
              cell: float = 0.25,
              north_axis: str = "y",
              east_axis: str = "x",
              up_axis: str = "z",
              surface_z: float = 0.0,
              bounds: Optional[Tuple[float, float, float, float]] = None,
              name: Optional[str] = None) -> TerrainField:
    """Resample a mesh onto a nav-frame depth grid.

    Args:
        path:       any mesh trimesh reads -- .dae, .obj, .ply, .stl.
        cell:       grid resolution (m).  Governs memory: a 500x500 m survey
                    at 0.25 m is 4 M cells, about 32 MB.
        north/east/up_axis: which mesh axis is which in nav.  Defaults suit
                    the bundled sawtooth.obj, whose header declares X = nav
                    east and Y = nav north.  Prefix with '-' to flip.
        surface_z:  mesh up-coordinate of the water surface; depth is measured
                    down from it.
        bounds:     (north0, north1, east0, east1) to clip a large survey.

    Interpolation reuses the mesh's own triangulation rather than
    re-triangulating, so the sampled surface is the reconstructed surface and
    holes stay holes (NaN) instead of being bridged.
    """
    import trimesh

    scene_or_mesh = trimesh.load(path, force="mesh")
    verts = np.asarray(scene_or_mesh.vertices, dtype=float)
    faces = np.asarray(scene_or_mesh.faces, dtype=int)
    if verts.size == 0 or faces.size == 0:
        raise ValueError(f"{path}: no geometry")

    def axis(spec):
        sign = -1.0 if spec.startswith("-") else 1.0
        idx = {"x": 0, "y": 1, "z": 2}[spec.lstrip("-").lower()]
        return sign, idx

    sn, an = axis(north_axis)
    se, ae = axis(east_axis)
    su, au = axis(up_axis)

    north = sn * verts[:, an]
    east = se * verts[:, ae]
    depth = surface_z - su * verts[:, au]      # positive down

    n0, n1 = north.min(), north.max()
    e0, e1 = east.min(), east.max()
    if bounds is not None:
        n0, n1, e0, e1 = bounds

    nn = max(2, int(math.ceil((n1 - n0) / cell)) + 1)
    ne = max(2, int(math.ceil((e1 - e0) / cell)) + 1)
    grid = np.full((nn, ne), np.nan, dtype=float)

    # Rasterise the triangles directly rather than interpolating over a planar
    # triangulation of the vertices.  matplotlib's Triangulation requires the
    # projection to be a valid 2-D mesh, which a survey reconstruction need
    # not be: a closed solid, an overhang, or two returns at one ground
    # position all project to overlapping or degenerate triangles and it
    # refuses the whole mesh.  Rasterising asks less -- each triangle is
    # filled independently and the shallowest surface wins, which is the
    # conservative reading for obstacle work.  Cells no triangle covers stay
    # NaN, so holes in the survey stay holes.
    ni = (north - n0) / cell
    ei = (east - e0) / cell
    for tri in faces:
        a, b, c = tri[0], tri[1], tri[2]
        ya, yb, yc = ni[a], ni[b], ni[c]
        xa, xb, xc = ei[a], ei[b], ei[c]
        i_lo = max(0, int(math.floor(min(ya, yb, yc))))
        i_hi = min(nn - 1, int(math.ceil(max(ya, yb, yc))))
        j_lo = max(0, int(math.floor(min(xa, xb, xc))))
        j_hi = min(ne - 1, int(math.ceil(max(xa, xb, xc))))
        if i_lo > i_hi or j_lo > j_hi:
            continue
        det = (yb - ya) * (xc - xa) - (yc - ya) * (xb - xa)
        if abs(det) < 1e-12:
            continue                      # edge-on in plan: contributes nothing
        II, JJ = np.meshgrid(np.arange(i_lo, i_hi + 1),
                             np.arange(j_lo, j_hi + 1), indexing="ij")
        w0 = ((yb - II) * (xc - JJ) - (yc - II) * (xb - JJ)) / det
        w1 = ((yc - II) * (xa - JJ) - (ya - II) * (xc - JJ)) / det
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        z = w0 * depth[a] + w1 * depth[b] + w2 * depth[c]
        sub = grid[i_lo:i_hi + 1, j_lo:j_hi + 1]
        cand = np.where(inside, z, np.nan)
        grid[i_lo:i_hi + 1, j_lo:j_hi + 1] = np.fmin(sub, cand)

    return TerrainField(grid, n0, e0, cell,
                        name or os.path.basename(path))


def _main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mesh", help="terrain mesh (.dae/.obj/.ply/.stl)")
    ap.add_argument("--cell", type=float, default=0.25)
    ap.add_argument("--north-axis", default="y")
    ap.add_argument("--east-axis", default="x")
    ap.add_argument("--up-axis", default="z")
    ap.add_argument("--surface-z", type=float, default=0.0)
    ap.add_argument("--leg", nargs=4, type=float, metavar=("N0", "E0", "N1", "E1"),
                    help="sample a leg and print its profile stats")
    ap.add_argument("--ideal", action="store_true",
                    help="also run the analytic in-band ceiling over the leg")
    ap.add_argument("--dump", metavar="CSV", help="write the leg profile to CSV")
    a = ap.parse_args(argv)

    field = from_mesh(a.mesh, cell=a.cell, north_axis=a.north_axis,
                      east_axis=a.east_axis, up_axis=a.up_axis,
                      surface_z=a.surface_z)
    print(field)

    if not a.leg:
        return 0

    fn = field.along((a.leg[0], a.leg[1]), (a.leg[2], a.leg[3]))
    L = fn.length
    s = np.arange(0.0, L, min(0.05, a.cell / 2))
    d = np.array([fn(float(v)) for v in s])
    good = np.isfinite(d)
    print(f"\nleg {L:.2f} m, {good.sum()}/{len(s)} samples on the survey")
    if good.any():
        print(f"  depth   {d[good].min():.2f} .. {d[good].max():.2f} m"
              f"   relief {d[good].max()-d[good].min():.2f} m")
        g = np.gradient(d[good], s[good])
        print(f"  |slope| max {np.nanmax(np.abs(g)):.2f}  "
              f"({math.degrees(math.atan(np.nanmax(np.abs(g)))):.1f} deg)")
    if a.dump:
        np.savetxt(a.dump, np.column_stack([s, d]), delimiter=",",
                   header="s_m,depth_m", comments="")
        print(f"  wrote {a.dump}")

    if a.ideal:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from ideal_trajectory import analytic_ideal
        # occupancy_map.py was retired in 577d6c2; the C++ core is the only
        # implementation, so there is nothing to fall back to.
        from occupancy_map_cpp import OccupancyMapConfig
        cfg = OccupancyMapConfig()
        res = analytic_ideal(fn, 0.0, L, cfg)
        print()
        print(res.report())
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
