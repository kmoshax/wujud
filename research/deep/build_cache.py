"""Builds research/cache/deep/{sim,real}_*.npy: gridded amplitude/phase/RSSI streams of simulated sessions
(from crates/app/examples/export_sim_windows.rs) and of every non-excluded ESPectre recording.

.venv/bin/python -m deep.build_cache [rooms]        (run from research/)
"""
import glob, json, os, shutil, subprocess, sys
from multiprocessing import Pool
import numpy as np
from deep.preprocess import gridded_streams, real_complex, sim_complex

CACHE = "cache/deep"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/.."
ESPECTRE = os.environ.get("ESPECTRE_DATA", os.path.expanduser("~/Projects/ext/espectre/data"))
SESSION_GRID = 1200
ROOMS_PER_BATCH = 300
CLASS_OF_ACTIVITY = {"away": 0, "lie": 1, "sit": 1, "stand": 1, "walk": 2}
CLASS_OF_LABEL = {"empty": 0, "static_presence": 1, "motion": 2}


def fixed_length(stream, length=SESSION_GRID):
    if len(stream) >= length:
        return stream[:length]
    pad = [(0, length - len(stream))] + [(0, 0)] * (stream.ndim - 1)
    return np.pad(stream, pad, mode="edge")


def process_sim_chunk(stem):
    iq = np.load(stem + ".iq.npy", mmap_mode="r")
    t = np.load(stem + ".t.npy").astype(np.float64)
    rssi = np.load(stem + ".rssi.npy")
    metas = json.load(open(stem + ".json"))
    amp, phase, rs, start = [], [], [], 0
    for meta in metas:
        stop = start + meta["packets"]
        a, p, r = gridded_streams(t[start:stop], sim_complex(np.asarray(iq[start:stop])), rssi[start:stop])
        amp.append(fixed_length(a)); phase.append(fixed_length(p)); rs.append(fixed_length(r))
        meta["label"] = CLASS_OF_ACTIVITY[meta["activity"]]
        start = stop
    return np.float16(amp), np.float16(phase), np.float16(rs), metas


def export_batch(first_room, rooms, out_dir):
    subprocess.run(["cargo", "run", "--release", "-q", "-p", "wujud", "--example", "export_sim_windows", "--",
                    os.path.abspath(out_dir), str(first_room), str(rooms), "60"], cwd=REPO, check=True)


def build_sim(total_rooms):
    raw = f"{CACHE}/sim_raw"
    amp, phase, rssi, metas = [], [], [], []
    for first in range(0, total_rooms, ROOMS_PER_BATCH):
        shutil.rmtree(raw, ignore_errors=True)
        export_batch(first, min(ROOMS_PER_BATCH, total_rooms - first), raw)
        stems = sorted(p[:-len(".json")] for p in glob.glob(f"{raw}/*.json"))
        with Pool(12) as pool:
            for a, p, r, m in pool.map(process_sim_chunk, stems):
                amp.append(a); phase.append(p); rssi.append(r); metas += m
        print(f"sim rooms {first + ROOMS_PER_BATCH}/{total_rooms}: {len(metas)} sessions", flush=True)
    shutil.rmtree(raw, ignore_errors=True)
    np.save(f"{CACHE}/sim_amp.npy", np.concatenate(amp))
    np.save(f"{CACHE}/sim_phase.npy", np.concatenate(phase))
    np.save(f"{CACHE}/sim_rssi.npy", np.concatenate(rssi))
    json.dump(metas, open(f"{CACHE}/sim_meta.json", "w"))


def process_real(job):
    label, meta = job
    z = np.load(os.path.join(ESPECTRE, label, meta["filename"]))
    t = np.unwrap(z["wifi_rx_ts_us"].astype(np.float64), period=2 ** 32) / 1e6
    order = np.argsort(t)
    a, p, r = gridded_streams(t[order], real_complex(z["csi_data"][order]), z["rssi_dbm"][order])
    info = dict(file=meta["filename"], label=CLASS_OF_LABEL[label], role=meta["dataset_role"], chip=meta["chip"],
                device=meta.get("device_id"), environment=meta.get("environment"), collected_at=meta.get("collected_at"),
                long=bool(meta.get("long_recording")), grid=len(a))
    return np.float16(a), np.float16(p), np.float16(r), info


def build_real():
    info = json.load(open(os.path.join(ESPECTRE, "dataset_info.json")))
    jobs = [(label, m) for label, files in info["files"].items() for m in files if m.get("dataset_role") != "exclude"]
    with Pool(8) as pool:
        out = pool.map(process_real, jobs)
    np.save(f"{CACHE}/real_amp.npy", np.concatenate([o[0] for o in out]))
    np.save(f"{CACHE}/real_phase.npy", np.concatenate([o[1] for o in out]))
    np.save(f"{CACHE}/real_rssi.npy", np.concatenate([o[2] for o in out]))
    json.dump([o[3] for o in out], open(f"{CACHE}/real_meta.json", "w"))
    print(f"real: {len(out)} recordings, {sum(o[3]['grid'] for o in out)} grid samples")


if __name__ == "__main__":
    os.makedirs(CACHE, exist_ok=True)
    build_real()
    build_sim(int(sys.argv[1]) if len(sys.argv) > 1 else 3000)
