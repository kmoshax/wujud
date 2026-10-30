"""3-D hall datasets (flat + tiered amphitheatres, nodes at mixed heights).

python -m hallsense.dataset3d --out data3/train --halls 80 --scenes 4 --seed 11 --nodes 6,8,10,12,16
python -m hallsense.dataset3d --out data3/sweep --halls 10 --seed 13 --node-sweep 4,6,8,12,16
"""
import argparse, json, os, time
from multiprocessing import Pool
import numpy as np

from .scenes3d import random_hall3
from .sim3d import simulate_hall3
from .sim import sample_occupancy


class _HallView:
    """adapter so sample_occupancy() can be reused"""
    def __init__(self, sc):
        self.seats = sc.seats[:, :2]; self.seat_rc = sc.seat_rc


def save(path, sc, res, extra):
    chest = sc.seats + np.array([0, 0, 1.0])
    np.savez_compressed(path, W=sc.W, D=sc.D, Hc=sc.H, nodes=sc.nodes, seats=chest, seat_rc=sc.seat_rc,
                        front=sc.tiers["front"], rise=sc.tiers["rise"], pitch=sc.tiers["pitch"], y0=sc.tiers["y0"],
                        aisles_x=np.array(sc.meta["aisles"]), links=np.array(sc.links), occ=res["occ"],
                        cal_mean=res["cal_mean"].astype(np.float16), cal_std=res["cal_std"].astype(np.float16),
                        ses_mean=res["ses_mean"].astype(np.float16), ses_std=res["ses_std"].astype(np.float16),
                        meta=json.dumps({"placement": sc.meta["placement"], "tiered": bool(sc.meta["tiered"]), **extra}))


def _job(args):
    out, seed, n_scenes, nodes_choice, sweep, harsh, restless, tiered = args
    rng = np.random.default_rng(seed)
    n = int(rng.choice(nodes_choice))
    sc = random_hall3(rng, n_nodes=n, tiered=tiered)
    done = []
    if sweep:
        base = random_hall3(np.random.default_rng(seed), n_nodes=4, tiered=tiered, placement="walls")
        occ, _ = sample_occupancy(_HallView(base), np.random.default_rng([seed, 99]))
        for k in sweep:
            s2 = random_hall3(np.random.default_rng(seed), n_nodes=k, tiered=tiered, placement="walls")
            res = simulate_hall3(s2, np.random.default_rng([seed, k]), occ)
            p = os.path.join(out, f"h{seed:06d}_n{k:02d}.npz")
            save(p, s2, res, {"hall_seed": seed, "n_nodes": k}); done.append(p)
        return done
    for s in range(n_scenes):
        srng = np.random.default_rng([seed, s])
        occ, pat = sample_occupancy(_HallView(sc), srng)
        res = simulate_hall3(sc, srng, occ, harsh=harsh, restless=restless)
        p = os.path.join(out, f"h{seed:06d}_s{s}.npz")
        save(p, sc, res, {"hall_seed": seed, "n_nodes": n, "pattern": str(pat), "harsh": harsh, "restless": restless}); done.append(p)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--halls", type=int, default=10)
    ap.add_argument("--scenes", type=int, default=2); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nodes", default="6,8,10,12,16"); ap.add_argument("--node-sweep", default="")
    ap.add_argument("--harsh", type=float, default=1.0); ap.add_argument("--restless", type=float, default=1.0)
    ap.add_argument("--tiered", default="random", choices=["random", "yes", "no"])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tiered = None if a.tiered == "random" else a.tiered == "yes"
    sweep = [int(x) for x in a.node_sweep.split(",")] if a.node_sweep else None
    jobs = [(a.out, a.seed * 100000 + h, a.scenes, [int(x) for x in a.nodes.split(",")], sweep, a.harsh, a.restless, tiered)
            for h in range(a.halls)]
    t0 = time.time(); n = 0
    with Pool(a.workers) as p:
        for d in p.imap_unordered(_job, jobs):
            n += len(d); print(f"[{time.time() - t0:7.1f}s] {n} scenes", flush=True)


if __name__ == "__main__":
    main()
