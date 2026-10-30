"""Benchmark on REAL ESP32 recordings from the ESPectre project (github.com/francescopace/espectre, GPLv3 data).
Labels: empty / static_presence (person present, still) / motion. ESPectre itself maps empty+static to IDLE.
Splits follow ESPectre's own dataset roles: train -> fit, selection+holdout -> test.
"""
import json, os, sys, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallsense.roomfeat import windows, amplitude_from_iq, FEAT_NAMES

ROOT = os.environ.get("ESPECTRE_DATA", "/home/claude/ext/espectre/data")
LAB = {"empty": 0, "static_presence": 1, "motion": 2}


def load_real(skip_long=True):
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    rows = []
    for lab, files in info["files"].items():
        for m in files:
            if m.get("dataset_role") == "exclude" or (skip_long and m.get("long_recording")):
                continue
            z = np.load(os.path.join(ROOT, lab, m["filename"]))
            iq = z["csi_data"].reshape(len(z["csi_data"]), -1, 2)
            t = z["wifi_rx_ts_us"].astype(np.float64)
            t = np.unwrap(t, period=2 ** 32) / 1e6
            order = np.argsort(t); t, iq = t[order], iq[order]
            amp = amplitude_from_iq(iq)
            X, rates = windows(t, amp, z["rssi_dbm"][order].astype(float))
            rows.append(dict(file=m["filename"], label=LAB[lab], role=m["dataset_role"], env=m.get("environment"),
                             chip=m["chip"], X=X, rates=rates, device=m.get("device_id")))
    return rows


if __name__ == "__main__":
    import joblib
    rows = load_real()
    os.makedirs("cache", exist_ok=True)
    joblib.dump(rows, "cache/espectre_feats.joblib")
    for r in rows[:3]:
        print(r["file"], r["X"].shape)
    print(len(rows), "recordings")
