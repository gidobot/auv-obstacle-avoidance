# SITL testing helpers

Thin wrappers around `docker exec` into the running `seeker-gazebo` container,
which is where `sitl_eval.py` has to run: it needs the LCM stack, and for
contacts it needs ROS. The container mounts this repo at `/oa` and the
acfr-lcm clone at `/acfr-lcm` (see `acfr_sim_ws/docker-compose.yaml`), so file
paths can be given in host form and are translated automatically — recordings
written under `runs/` land on the host with no `docker cp` either way.

All of them find the container by name; set `SIM_CONTAINER` to override, and
`VEHICLE` if the LCM vehicle name is not `SEEKER-SITL`.

| script | what it is for |
| --- | --- |
| `sim_status.sh` | one-shot health check before trusting a run |
| `sim_watch.sh` | live progress in the terminal, ssh friendly |
| `sim_serve.sh` | web viewer: planned mission underlay + live track |
| `sim_record.sh` | record a run to `.npz` for scoring |

## Check before you trust a run

```bash
./sim_status.sh
```

Reports whether Gazebo is alive, whether the planner stack is up, whether
ground truth is publishing, and whether anything is publishing `ACFR_NAV`
twice. Each of those has silently invalidated a run at least once: a mission
can appear to run for an hour while the sim underneath it is dead, and two
publishers on `ACFR_NAV` drove the mapper's along-track coordinate 112x
faster than the vehicle actually moved.

## Watch a mission

```bash
./sim_serve.sh ../../../acfr-lcm/missions/simulator/lawnmower_sim_sawtooth_exp.xml
# then open http://localhost:8095/
```

or, with no display available:

```bash
./sim_watch.sh
```

Both warn when distance to the goal stops falling, which is the live signature
of a planner replanning the vehicle back behind itself, and of a leg that will
never arrive. Catching that during a run rather than afterwards is the whole
reason these exist.

## Record and score

```bash
./sim_record.sh ../../runs/new.npz --label cliff-manifold      # ctrl-C to stop
cd ../.. && python sitl_eval.py report runs/old.npz runs/new.npz
python sitl_eval.py plot runs/new.npz --out traj.png
```

Recording runs until interrupted and autosaves every 60 s, so a hard kill
costs at most the last minute.
