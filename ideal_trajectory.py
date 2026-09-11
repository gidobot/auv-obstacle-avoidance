"""Analytic ceiling on in-band survey line for a 1-D terrain profile.

The objective of a fixed-altitude survey is *in-band line*: the distance of the
transect flown within a tight band of the imaging altitude.  The terrain sets a
ceiling on it, and because the optimal policy has a closed characterisation the
ceiling can be computed from the profile rather than searched for.

The characterisation (see the positioning document, section 5): any metre flown
while outside the band is transect not surveyed, so forward motion out of band
is justified only where it is *compulsory*.  Correcting in place costs time but
no line, because a stationary vehicle covers no ground.  Three things are
compulsory:

  unfollowable faces   A vehicle at forward speed v_x tracking gradient g must
                       change depth at g * v_x, so the band can be held while
                       moving only where g <= v_z / v_x.  Steeper ground must
                       be crossed out of band.

  approach standoff    The vehicle must reach clearance depth before arriving
                       at a rising face, so it is above the band for the
                       standoff preceding it.

  tail clearance       After a descending edge the vehicle may not follow the
                       terrain down until its stern has crossed, so it is above
                       the band for half a vehicle length plus standoff.

Everything else is followable and can be flown in band, so

    ideal in-band line  =  transect length  -  |union of compulsory intervals|

Assumptions, which belong in the paper wherever the ceiling is quoted:

  * The transect is fixed, not the mission duration.  A time-limited mission
    would trade differently, since halting costs time.
  * Correcting in place is always available — this is where hover capability
    enters as a premise rather than a convenience.
  * Compulsory out-of-band motion is exactly the three cases above.  An ascent
    to clear a crest counts as compulsory, at the safety standoff.
  * The vehicle flies at one forward speed or none.  A controller free to creep
    at v_z / g could hold the band across a steep face, at a survey rate that
    would make the imagery useless; that option is excluded by the platform.
"""

import numpy as np

__all__ = ["IdealProfile", "analytic_ideal"]


class IdealProfile:
    """Result of an analytic ideal computation."""

    def __init__(self, length, in_band, lost, intervals, dx):
        self.length = length            # transect length (m)
        self.in_band = in_band          # ideal in-band line (m)
        self.lost = lost                # dict: cause -> metres lost
        self.intervals = intervals      # list of (start, end, cause)
        self.dx = dx

    @property
    def fraction(self):
        return self.in_band / self.length if self.length else float("nan")

    def __repr__(self):
        parts = "  ".join(f"{k} {v:.1f}" for k, v in sorted(self.lost.items()))
        return (f"IdealProfile({self.in_band:.1f}/{self.length:.1f} m = "
                f"{100*self.fraction:.1f}%  lost: {parts})")

    def report(self):
        out = [f"transect            {self.length:8.1f} m",
               f"ideal in-band line  {self.in_band:8.1f} m  ({100*self.fraction:.1f}%)",
               "compulsory out-of-band:"]
        for k, v in sorted(self.lost.items(), key=lambda kv: -kv[1]):
            out.append(f"   {k:<18} {v:8.1f} m  ({100*v/self.length:.1f}%)")
        return "\n".join(out)


def _merge(intervals, x0, x1):
    """Union of half-open intervals, clipped to [x0, x1], keeping causes.

    Overlaps are attributed to the first cause encountered so the breakdown
    sums to the total rather than double-counting.
    """
    intervals = sorted(intervals, key=lambda iv: iv[0])
    merged, claimed = [], []
    for a, b, cause in intervals:
        a, b = max(a, x0), min(b, x1)
        if b <= a:
            continue
        # subtract whatever is already claimed
        pieces = [(a, b)]
        for ca, cb in claimed:
            nxt = []
            for pa, pb in pieces:
                if cb <= pa or ca >= pb:
                    nxt.append((pa, pb))
                else:
                    if pa < ca:
                        nxt.append((pa, ca))
                    if cb < pb:
                        nxt.append((cb, pb))
            pieces = nxt
        for pa, pb in pieces:
            if pb > pa:
                merged.append((pa, pb, cause))
                claimed.append((pa, pb))
    return merged


def analytic_ideal(terrain_fn, x0, x1, cfg, dx=0.05, standoff=None):
    """Ceiling on in-band line for `terrain_fn` over [x0, x1].

    Args:
        terrain_fn: f(x) -> seabed depth (m, positive down).
        x0, x1:     transect extent (m).
        cfg:        OccupancyMapConfig — supplies the speeds, vehicle length and
                    safety standoff.
        dx:         sampling step for the gradient scan (m).
        standoff:   horizontal safety standoff (m); defaults to
                    cfg.safety_standoff_m.

    Returns:
        IdealProfile.
    """
    if standoff is None:
        standoff = cfg.safety_standoff_m
    g_max = cfg.vertical_speed / max(cfg.survey_speed, 1e-9)
    half = cfg.vehicle_length / 2.0

    x = np.arange(x0, x1 + dx, dx)
    z = np.array([terrain_fn(float(v)) for v in x], dtype=float)
    # depth is positive down, so a *rising* seabed has decreasing z
    grad = np.gradient(z, dx)

    intervals = []

    # Runs are grouped by the *sign* of the gradient, not merely by steepness.
    # A rising face and the drop that follows it are adjacent, so grouping on
    # steepness alone merges them into one run whose endpoints look level —
    # which silently loses the approach standoff belonging to the rise.
    rising_mask  = grad < -g_max      # depth decreasing => seabed rising
    falling_mask = grad >  g_max      # depth increasing => seabed falling

    def runs(mask):
        out, i = [], 0
        while i < len(mask):
            if not mask[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(mask) and mask[j + 1]:
                j += 1
            out.append((x[i], x[j]))
            i = j + 1
        return out

    for a, b in runs(rising_mask):
        # the vehicle must be at clearance depth on arrival, so it is above the
        # band for the standoff preceding the face
        intervals.append((a - standoff, a, "approach standoff"))
        intervals.append((a, b, "unfollowable face"))

    for a, b in runs(falling_mask):
        intervals.append((a, b, "unfollowable face"))
        # the stern must cross before the vehicle may follow the terrain down
        intervals.append((b, b + half + standoff, "tail clearance"))

    merged = _merge(intervals, x0, x1)
    lost = {}
    for a, b, cause in merged:
        lost[cause] = lost.get(cause, 0.0) + (b - a)
    total_lost = sum(lost.values())
    length = x1 - x0
    return IdealProfile(length, max(0.0, length - total_lost), lost, merged, dx)


if __name__ == "__main__":
    import math
    import occupancy_map_cpp as cpp
    from compare_controllers import SAWTOOTH, LAWNMOWER
    from simulator import make_terrain_3d_sawtooth

    cfg = cpp.OccupancyMapConfig()
    terrain3d = make_terrain_3d_sawtooth(**SAWTOOTH)
    profile = lambda x: terrain3d(x, 0.0)
    leg = LAWNMOWER["leg_length"]

    print(f"followable gradient  {cfg.vertical_speed/cfg.survey_speed:g} "
          f"({math.degrees(math.atan(cfg.vertical_speed/cfg.survey_speed)):.0f} deg)")
    print(f"standoff             {cfg.safety_standoff_m:g} m")
    print(f"vehicle length       {cfg.vehicle_length:g} m\n")
    res = analytic_ideal(profile, 0.0, leg, cfg)
    print(res.report())
