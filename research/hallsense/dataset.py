"""Generate simulated lecture-hall datasets in parallel.

python -m hallsense.dataset --out data/train --halls 90 --scenes 4 --seed 1
Each scene -> one .npz with hall geometry, per-link window features, seat labels.
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np

from .geometry import random_hall, Hall
from .sim import simulate_scene, sample_occupancy


def save_scene(path, hall: Hall, res, extra):
    np.savez_compressed(
        path,
        W=hall.W, D=hall.D, Hc=hall.Hc, nodes=hall.nodes, node_h=hall.node_h, seats=hall.seats,
        seat_rc=hall.seat_rc, front=hall.front, aisles_x=np.array(hall.aisles_x),
        links=np.array(hall.links), occ=res["occ"],
        cal_mean=res["cal_mean"].astype(np.float16), cal_std=res["cal_std"].astype(np.float16),
        ses_mean=res["ses_mean"].astype(np.float16), ses_std=res["ses_std"].astype(np.float16),
        ses_n=res["ses_n"].astype(np.float16),
        meta=json.dumps({"pattern": str(res["pattern"]), "n_walkers": int(res["n_walkers"]),
                         "placement": hall.placement, **extra}))


def load_hall(z):
    return Hall(float(z["W"]), float(z["D"]), float(z["Hc"]), z["nodes"], z["node_h"], z["seats"],
                z["seat_rc"], np.zeros(4), float(z["front"]), list(z["aisles_x"]), "")


def _job(args):
    out, hall_seed, n_scenes, nodes_choice, placement, harsh, restless, node_sweep, shared_env = args
    rng = np.random.default_rng(hall_seed)
    n_nodes = int(rng.choice(nodes_choice))
    hall = random_hall(rng, n_nodes=n_nodes, placement=placement)
    done = []
    for s in range(n_scenes):
        srng = np.random.default_rng([hall_seed, s])
        if node_sweep:
            # same hall size + same occupancy, different number of nodes
            occ, _ = sample_occupancy(hall, np.random.default_rng([hall_seed, 999]))
            for n in node_sweep:
                h2 = random_hall(np.random.default_rng([hall_seed, 7, n]), n_nodes=n, placement=placement or "perimeter",
                                 W=hall.W, D=hall.D)
                # keep the seat grid of the base hall so occupancy is comparable
                h2.seats, h2.seat_rc, h2.front, h2.aisles_x = hall.seats, hall.seat_rc, hall.front, hall.aisles_x
                res = simulate_scene(h2, np.random.default_rng([hall_seed, s, n]), occ=occ, harsh=harsh, restless=restless)
                p = os.path.join(out, f"h{hall_seed:05d}_s{s}_n{n:02d}.npz")
                save_scene(p, h2, res, {"hall_seed": hall_seed, "n_nodes": n, "harsh": harsh, "restless": restless})
                done.append(p)
            break
        res = simulate_scene(hall, srng, harsh=harsh, restless=restless, env_seed=hall_seed if shared_env else None)
        p = os.path.join(out, f"h{hall_seed:05d}_s{s}.npz")
        save_scene(p, hall, res, {"hall_seed": hall_seed, "n_nodes": n_nodes, "harsh": harsh, "restless": restless})
        done.append(p)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--halls", type=int, default=10)
    ap.add_argument("--scenes", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nodes", default="8,10,12,14,16,20")
    ap.add_argument("--placement", default=None, choices=[None, "perimeter", "perimeter+aisle", "rows"])
    ap.add_argument("--harsh", type=float, default=1.0, help=">1 = worse ESP32 impairments")
    ap.add_argument("--restless", type=float, default=1.0, help="<1 = calmer crowd (fewer fidgets)")
    ap.add_argument("--node-sweep", default="", help="e.g. 6,8,12,16,24 (same hall/occupancy)")
    ap.add_argument("--shared-env", action="store_true", help="scenes of one hall = lectures in the same deployment")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nodes = [int(x) for x in a.nodes.split(",")]
    sweep = [int(x) for x in a.node_sweep.split(",")] if a.node_sweep else None
    jobs = [(a.out, a.seed * 100000 + h, a.scenes, nodes, a.placement, a.harsh, a.restless, sweep, a.shared_env) for h in range(a.halls)]
    t0 = time.time()
    n = 0
    with Pool(a.workers) as pool:
        for d in pool.imap_unordered(_job, jobs):
            n += len(d)
            print(f"[{time.time() - t0:7.1f}s] {n} scenes", flush=True)


if __name__ == "__main__":
    main()
