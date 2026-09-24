#!/usr/bin/env python3
"""Record and score SITL runs, for comparing obstacle-avoidance controllers.

The comparison the paper needs is between the deployed cliff-manifold method
(`oa-mapper`, publishing OA_COMMAND) and the predecessor that actually flew
these missions (`oa-processor`, publishing OA).  Both are driven through the
same local-planner over the same terrain, so the honest comparison is of the
*vehicle trajectory*, not of either controller's internal state.  This tool
therefore records ground truth and scores it, and treats controller output as
annotation.

Three subcommands:

    record   inside the seeker-gazebo container; writes a .npz
    report   anywhere with numpy; prints the metrics table
    plot     anywhere with numpy+matplotlib; 3-D trajectory figure

Recording and analysis are split because recording needs the sim's ROS and LCM
stacks while analysis needs neither -- a .npz can be scored and re-scored on a
laptop, with a different altitude band, long after the run.

WHAT IS MEASURED, AND WHY

in-band line   Distance of the *lateral* (horizontal) trajectory flown within
               `--band` of the imaging altitude.  Lateral, not elapsed time:
               time spent correcting in place costs no survey line, which is
               the whole premise of the deployed policy, and a time-based
               metric would score a hovering vehicle the same as a surveying
               one.  This is the headline number.

collisions     Read from the physics engine via the hull contact sensor, not
               inferred from altitude.  The nadir altimeter cannot see a bow or
               stern strike on a slope -- precisely the hazard the method
               exists to prevent -- so an altitude-derived collision count
               would be blind to its own failure mode.  Episodes are counted,
               not samples, so one scrape is one collision.

worst clearance
               Minimum nadir altitude over the run.  Reported as a severity
               hint alongside the contact count, NOT as a collision test: it
               is nadir-only for the reason above.

All arrays are the same length and aligned to the ground-truth nav samples, so
trajectory, altitude, band membership and contact state can be plotted against
one another directly.
"""

import argparse
import json
import math
import sys

import numpy as np

SCHEMA = 1


# ── recording (needs the sim's ROS + LCM stacks) ─────────────────────────────

def cmd_record(args):
    import importlib.util as ilu
    import os
    import threading
    import time

    import lcm

    def load(pkg, name):
        """Load a generated LCM type.

        Prefer the acfr-lcm build tree when it is mounted, because that is the
        only place the OA types are generated.  Fall back to the installed
        package, which is what the seeker-gazebo image ships -- enough for the
        nav types, so a run can be recorded there without mounting anything.
        """
        path = os.path.join(args.lcmtypes, pkg, name + ".py")
        if os.path.exists(path):
            spec = ilu.spec_from_file_location(name, path)
            mod = ilu.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return getattr(mod, name)
        return getattr(__import__(pkg, fromlist=[name]), name)

    nav_t = load("acfrlcm", "auv_acfr_nav_t")
    try:
        oa_cmd_t = load("acfrlcm", "auv_oa_command_t")
    except Exception:
        oa_cmd_t = None

    V = args.vehicle
    # Latest-value cache; every channel but the trigger is sampled, not queued,
    # so the record stays on one clock (the ground-truth nav rate).
    # `contact_utime` rather than a boolean: the Contact system publishes only
    # while surfaces touch and sends nothing at all when clear, so a latched
    # flag would stay true forever after the first strike -- one endless
    # "collision" for the rest of the run.  Contact is therefore defined as
    # "a contacts message arrived within CONTACT_HOLD_S", which decays on its
    # own when the messages stop.
    CONTACT_HOLD_S = 0.3
    latest = {"mode": "", "oa_vx": float("nan"), "oa_target": float("nan"),
              "contact_t": -1e9, "n_contact": 0, "filt_heading": float("nan")}
    rows = []
    stop = threading.Event()

    def on_gt(ch, data):
        m = nav_t.decode(data)
        now = time.time()
        touching = 1 if (now - latest["contact_t"]) <= CONTACT_HOLD_S else 0
        rows.append((now, m.x, m.y, m.depth, m.altitude, m.heading,
                     m.vx, touching, latest["n_contact"] if touching else 0,
                     latest["mode"], latest["oa_vx"], latest["oa_target"],
                     latest["filt_heading"]))

    def on_filt(ch, data):
        latest["filt_heading"] = nav_t.decode(data).heading

    def on_oa_cmd(ch, data):
        if oa_cmd_t is None:
            return
        m = oa_cmd_t.decode(data)
        latest["mode"] = m.control_mode
        latest["oa_vx"] = m.vx
        latest["oa_target"] = m.vertical_target

    lc = lcm.LCM()
    lc.subscribe(f"{V}_GT.ACFR_NAV", on_gt)
    lc.subscribe(f"{V}.ACFR_NAV", on_filt)
    lc.subscribe(f"{V}.OA_COMMAND", on_oa_cmd)

    # Contacts come over ROS: they are a simulator fact with no LCM channel, and
    # inventing one would mean new message types in two more repos.
    ros_ctx = None
    if not args.no_contacts:
        try:
            import rclpy
            from rclpy.node import Node
            from ros_gz_interfaces.msg import Contacts

            rclpy.init(args=None)
            node = Node("sitl_eval_recorder")

            def on_contacts(msg):
                n = len(msg.contacts)
                if n:
                    latest["contact_t"] = time.time()
                    latest["n_contact"] = n

            node.create_subscription(Contacts, args.contact_topic, on_contacts, 10)

            def spin():
                while not stop.is_set():
                    rclpy.spin_once(node, timeout_sec=0.1)
            ros_thread = threading.Thread(target=spin, daemon=True)
            ros_thread.start()
            ros_ctx = (node, ros_thread)
            print(f"contacts: subscribed to {args.contact_topic}")
        except Exception as exc:
            print(f"contacts: UNAVAILABLE ({exc}) -- recording without them",
                  file=sys.stderr)

    def write_npz():
        """Serialise whatever has been recorded so far."""
        if not rows:
            return
        snap = list(rows)                      # the LCM callback may still append
        modes = sorted({r[9] for r in snap})
        mode_ix = {m: i for i, m in enumerate(modes)}
        a = lambda i, dt=float: np.array([r[i] for r in snap], dtype=dt)
        np.savez_compressed(
            args.out,
            schema=SCHEMA,
            t=a(0) - snap[0][0], x=a(1), y=a(2), depth=a(3), altitude=a(4),
            heading=a(5), vx=a(6), contact=a(7, np.int8), n_contact=a(8, np.int16),
            mode=np.array([mode_ix[r[9]] for r in snap], dtype=np.int16),
            mode_names=np.array(modes, dtype=object),
            oa_vx=a(10), oa_target=a(11), filter_heading=a(12),
            meta=json.dumps({"vehicle": V, "label": args.label,
                             "duration_s": snap[-1][0] - snap[0][0],
                             "samples": len(snap),
                             "contacts_recorded": not args.no_contacts}),
        )

    # SIGTERM as well as SIGINT: a long run is more likely to be stopped with
    # `docker stop` or `kill` than with ctrl-C, and an unwritten .npz would
    # throw the whole run away.
    import signal
    halt = {"now": False}

    def _stop(signum, frame):
        halt["now"] = True
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, _stop)
        except (ValueError, OSError):
            pass

    unlimited = args.duration <= 0
    print(f"recording {V} "
          + ("until interrupted" if unlimited else f"for {args.duration:.0f}s")
          + " ... (ctrl-C, or SIGTERM, to stop and write)")
    t0 = time.time()
    last_report = last_save = t0
    try:
        while not halt["now"] and (unlimited or time.time() - t0 < args.duration):
            try:
                lc.handle_timeout(200)
            except OSError:
                # A signal interrupts the poll inside lcm_handle_timeout, which
                # surfaces as OSError.  Python may not have run the handler yet
                # when the exception propagates, so yield briefly and re-check
                # before deciding this is a real transport error.
                time.sleep(0.05)
                if halt["now"]:
                    break
                raise
            now = time.time()
            if now - last_report >= args.progress > 0:
                last_report = now
                band = sum(1 for r in rows
                           if r[4] == r[4] and abs(r[4] - args.altitude) <= args.band)
                touch = sum(1 for r in rows if r[7])
                print(f"  {now-t0:6.0f}s  {len(rows):6d} samples  "
                      f"in band {100*band/max(len(rows),1):5.1f}%  "
                      f"contact samples {touch}", flush=True)
            if args.autosave > 0 and now - last_save >= args.autosave:
                last_save = now
                write_npz()
    except KeyboardInterrupt:
        pass
    if halt["now"]:
        print("\nstopping on signal")
    stop.set()

    # Shut ROS down deliberately.  Leaving the spin thread running as a daemon
    # lets the interpreter tear rclpy down underneath it, which segfaults on
    # exit -- after the data is written, but not something to rely on.
    if ros_ctx is not None:
        node, ros_thread = ros_ctx
        ros_thread.join(timeout=2.0)
        try:
            import rclpy
            node.destroy_node()
            rclpy.shutdown()
        except Exception:
            pass

    if not rows:
        print(f"NO DATA on {V}_GT.ACFR_NAV -- is PUBLISH_GT_NAV=true?",
              file=sys.stderr)
        return 1

    write_npz()
    print(f"wrote {args.out}  ({len(rows)} samples, {time.time()-t0:.0f}s)")
    return 0


# ── scoring (pure numpy) ─────────────────────────────────────────────────────

class Run:
    """A recorded run, with the derived quantities the metrics need."""

    def __init__(self, path, band, target_alt):
        d = np.load(path, allow_pickle=True)
        self.path = path
        self.meta = json.loads(str(d["meta"]))
        self.label = self.meta.get("label") or path
        self.t = d["t"]
        self.x, self.y, self.depth = d["x"], d["y"], d["depth"]
        self.altitude, self.heading = d["altitude"], d["heading"]
        self.contact = d["contact"].astype(bool)
        self.mode = d["mode"]
        self.mode_names = list(d["mode_names"])
        self.band, self.target_alt = band, target_alt

        # Lateral step between consecutive samples: the survey line is ground
        # covered, so vertical motion must not count toward it.
        self.step = np.hypot(np.diff(self.x, prepend=self.x[0]),
                             np.diff(self.y, prepend=self.y[0]))
        self.valid_alt = np.isfinite(self.altitude) & (self.altitude > 0)
        self.in_band = self.valid_alt & (np.abs(self.altitude - target_alt) <= band)

    @property
    def lateral(self):
        return float(self.step.sum())

    @property
    def in_band_line(self):
        return float(self.step[self.in_band].sum())

    @property
    def pct_in_band(self):
        return 100.0 * self.in_band_line / self.lateral if self.lateral else float("nan")

    @property
    def pct_time_in_band(self):
        return 100.0 * self.in_band.sum() / len(self.in_band) if len(self.in_band) else float("nan")

    def episodes(self, mask):
        """Contiguous True runs -> (count, total lateral distance)."""
        m = mask.astype(np.int8)
        starts = int(np.sum((m[1:] == 1) & (m[:-1] == 0))) + int(m[0] == 1)
        return starts, float(self.step[mask].sum())

    @property
    def collisions(self):
        return self.episodes(self.contact)

    def mode_breakdown(self):
        out = {}
        for i, name in enumerate(self.mode_names):
            if not name:
                continue
            sel = self.mode == i
            if sel.any():
                out[name] = float(self.step[sel].sum())
        return out

    def summary(self):
        n_coll, coll_dist = self.collisions
        alt = self.altitude[self.valid_alt]
        return {
            "label": self.label,
            "lateral_m": self.lateral,
            "in_band_m": self.in_band_line,
            "pct_in_band": self.pct_in_band,
            "pct_time_in_band": self.pct_time_in_band,
            "collisions": n_coll,
            "collision_m": coll_dist,
            "min_altitude": float(alt.min()) if alt.size else float("nan"),
            "median_altitude": float(np.median(alt)) if alt.size else float("nan"),
            "duration_s": float(self.t[-1]) if self.t.size else 0.0,
        }


def cmd_report(args):
    runs = [Run(p, args.band, args.altitude) for p in args.runs]
    rows = [r.summary() for r in runs]
    print(f"\ntarget altitude {args.altitude:g} m, band +/-{args.band:g} m\n")
    hdr = (f"{'run':<22} {'lateral':>9} {'in-band':>9} {'% line':>8} "
           f"{'% time':>8} {'colls':>6} {'coll m':>8} {'min alt':>8}")
    print(hdr); print("-" * len(hdr))
    for s in rows:
        print(f"{s['label'][:22]:<22} {s['lateral_m']:9.1f} {s['in_band_m']:9.1f} "
              f"{s['pct_in_band']:7.1f}% {s['pct_time_in_band']:7.1f}% "
              f"{s['collisions']:6d} {s['collision_m']:8.2f} {s['min_altitude']:8.2f}")
    for r in runs:
        mb = r.mode_breakdown()
        if mb:
            print(f"\n{r.label}: lateral distance by control mode")
            for k, v in sorted(mb.items(), key=lambda kv: -kv[1]):
                print(f"   {k:<18} {v:8.1f} m  ({100*v/r.lateral:5.1f}%)")
    if len(runs) == 2:
        a, b = rows
        print(f"\n{b['label']} vs {a['label']}:")
        d = b['pct_in_band'] - a['pct_in_band']
        print(f"   in-band line   {d:+.1f} points ({b['in_band_m']-a['in_band_m']:+.1f} m)")
        print(f"   collisions     {b['collisions']-a['collisions']:+d}")
    if args.json:
        open(args.json, "w").write(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json}")
    return 0


def cmd_plot(args):
    try:
        import matplotlib
        matplotlib.use("Agg" if args.out else matplotlib.get_backend())
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D            # noqa: F401
    except ImportError:
        print("matplotlib required:  pip install matplotlib", file=sys.stderr)
        return 1

    r = Run(args.run, args.band, args.altitude)
    fig = plt.figure(figsize=(13, 5.5))

    ax = fig.add_subplot(121, projection="3d")
    # depth is positive down; negate so the figure reads the way a diver would
    z = -r.depth
    ok = r.in_band & ~r.contact
    out = ~r.in_band & ~r.contact
    ax.plot(r.x, r.y, z, color="0.8", lw=0.8, zorder=1)
    ax.scatter(r.x[out], r.y[out], z[out], s=4, c="#B1187A", label="out of band")
    ax.scatter(r.x[ok], r.y[ok], z[ok], s=4, c="#1F6B4A", label="in band")
    if r.contact.any():
        ax.scatter(r.x[r.contact], r.y[r.contact], z[r.contact], s=42,
                   marker="X", c="#D7263D", edgecolor="k", linewidth=0.4,
                   label=f"contact ({r.collisions[0]})", zorder=5)
    ax.set_xlabel("north (m)"); ax.set_ylabel("east (m)"); ax.set_zlabel("z (m)")
    ax.set_title(f"{r.label}\n{r.pct_in_band:.1f}% of lateral line in band")
    ax.legend(loc="upper left", fontsize=8)

    ax2 = fig.add_subplot(122)
    ax2.axhspan(r.target_alt - r.band, r.target_alt + r.band,
                color="#1F6B4A", alpha=0.15, label=f"band +/-{r.band:g} m")
    ax2.plot(r.t, np.where(r.valid_alt, r.altitude, np.nan), lw=1.0,
             color="#2A6A88", label="altitude")
    if r.contact.any():
        ax2.scatter(r.t[r.contact], r.altitude[r.contact], s=36, marker="X",
                    c="#D7263D", zorder=5, label="contact")
    ax2.set_xlabel("time (s)"); ax2.set_ylabel("altitude (m)")
    ax2.set_title("altitude vs the imaging band")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)

    fig.tight_layout()
    if args.out:
        fig.savefig(args.out, dpi=140)
        print(f"wrote {args.out}")
    else:
        plt.show()
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--altitude", type=float, default=2.0,
                       help="target imaging altitude (m)")
        p.add_argument("--band", type=float, default=0.5,
                       help="half-width of the imaging band (m)")

    r = sub.add_parser("record", help="record a run (inside seeker-gazebo)")
    r.add_argument("out")
    r.add_argument("--vehicle", default="SEEKER-SITL")
    r.add_argument("--duration", type=float, default=0.0,
                   help="seconds; 0 or less records until interrupted (default)")
    r.add_argument("--progress", type=float, default=30.0,
                   help="seconds between progress lines; 0 to silence")
    r.add_argument("--autosave", type=float, default=60.0,
                   help="seconds between intermediate writes; 0 to disable")
    r.add_argument("--altitude", type=float, default=2.0,
                   help="target altitude, for the progress line only")
    r.add_argument("--band", type=float, default=0.5,
                   help="band half-width, for the progress line only")
    r.add_argument("--label", default="", help="name for the report table")
    r.add_argument("--contact-topic", default="/seeker/contacts")
    r.add_argument("--no-contacts", action="store_true")
    r.add_argument("--lcmtypes",
                   default="/root/git/acfr-lcm/build/lib/python3.8/dist-packages/perls/lcmtypes")
    r.set_defaults(func=cmd_record)

    p = sub.add_parser("report", help="score one or more recordings")
    p.add_argument("runs", nargs="+")
    p.add_argument("--json", help="also write the summary as JSON")
    common(p)
    p.set_defaults(func=cmd_report)

    q = sub.add_parser("plot", help="3-D trajectory + altitude figure")
    q.add_argument("run")
    q.add_argument("--out", help="write to a file instead of showing")
    common(q)
    q.set_defaults(func=cmd_plot)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
