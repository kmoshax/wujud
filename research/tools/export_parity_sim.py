"""Raw-output statistics of the Python room simulator, for the Rust statistical-parity test.

Run from research/:  .venv/bin/python tools/export_parity_sim.py [--rooms 30] [--workers 12]
Writes crates/sim/tests/golden/parity_room.json: per activity, one value per room for each statistic
(the same recipe as room_sim2real: random room, smart placement 30 %, fs ~ U(50,120), harsh ~ U(0.8,1.6),
60 s of one activity), plus wall-clock timings of the simulator.
"""
import argparse, json, os, sys, time
from multiprocessing import Pool
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallsense.scenes3d import random_room, ACTIVITIES
from hallsense.sim3d import simulate_room

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "crates", "sim", "tests", "golden", "parity_room.json")
SECONDS = 60


def link_stats(o):
    """the statistics also computed by crates/sim/tests/parity.rs::link_stats"""
    iq = o["iq"].astype(np.float64)
    amp = np.sqrt(iq[..., 0] ** 2 + iq[..., 1] ** 2)
    norm = amp / np.maximum(amp.mean(1, keepdims=True), 1e-6)
    return dict(
        amplitude_level=float(amp.mean()),
        temporal_std=float(np.median(norm.std(0))),
        rssi_median=float(np.median(o["rssi"])),
        rssi_std=float(o["rssi"].astype(float).std()),
        kept_fraction=float(len(o["t"]) / int(SECONDS * o["fs"])),
    )


def one_room(seed):
    rng = np.random.default_rng(seed)
    sc = random_room(rng, smart=rng.random() < 0.3)
    fs = rng.uniform(50, 120)
    harsh = rng.uniform(0.8, 1.6)
    res = {}
    t0 = time.time()
    for k, act in enumerate(ACTIVITIES):
        out, lab, _ = simulate_room(sc, np.random.default_rng([seed, k]), seconds=SECONDS, fs=fs, plan=[(act, SECONDS)], harsh=harsh)
        o = dict(out[0]); o["fs"] = fs
        res[act] = link_stats(o)
    return res, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rooms", type=int, default=30)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--single", type=int, default=3, help="rooms timed single-process")
    a = ap.parse_args()
    t0 = time.time()
    single = [one_room(10_000 + s)[1] for s in range(a.single)]
    single_s_per_room_minute = float(np.sum(single) / (a.single * len(ACTIVITIES)))
    t0 = time.time()
    with Pool(a.workers) as p:
        rooms = p.map(one_room, range(a.rooms))
    wall = time.time() - t0
    stats = {act: {name: [r[0][act][name] for r in rooms] for name in rooms[0][0][act]} for act in ACTIVITIES}
    quant = {act: {name: np.quantile(v, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist() for name, v in s.items()} for act, s in stats.items()}
    json.dump(dict(seconds=SECONDS, rooms=a.rooms, per_room=stats, quantiles_10_25_50_75_90=quant,
                   timing=dict(single_process_s_per_room_minute=single_s_per_room_minute,
                               pool_workers=a.workers, pool_s_per_room_minute=wall / (a.rooms * len(ACTIVITIES)),
                               pool_wall_s=wall, cpu=os.cpu_count())),
              open(OUT, "w"), indent=1)
    print(json.dumps(quant["walk"]), json.dumps(quant["away"]))
    print(f"single {single_s_per_room_minute:.3f} s/room-min, pool {wall / (a.rooms * len(ACTIVITIES)):.3f} s/room-min")


if __name__ == "__main__":
    main()
