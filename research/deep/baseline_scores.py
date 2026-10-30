"""Per-window still-vs-empty scores of the hand-feature baseline (cache/phase_exp_amplitude_current.joblib, the
'amplitude (current)' model of bench/phase_exp.py) on selection and holdout, for paired bootstrap comparisons.

.venv/bin/python -m deep.baseline_scores
"""
import json, os, sys
from multiprocessing import Pool
import joblib
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench"))
from phase_exp import ROOT, _real, metrics

if __name__ == "__main__":
    bundle = joblib.load("cache/phase_exp_amplitude_current.joblib")
    model, cols = bundle["model"], bundle["cols"]
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    jobs = [(lab, m) for lab, fs in info["files"].items() for m in fs if m.get("dataset_role") in ("selection", "holdout")]
    with Pool(8) as pool:
        real = pool.map(_real, jobs)
    saved, summary = {}, {}
    for split in ("selection", "holdout"):
        rows = [r for r in real if r["role"] == split]
        summary[split] = metrics(model, rows, cols)
        for r in rows:
            prob = model.predict_proba(r["X"][:, cols])
            saved[f"{split}/{r['file']}/probabilities"] = prob
            saved[f"{split}/{r['file']}/still"] = prob[:, 1] / (prob[:, 0] + prob[:, 1] + 1e-9)
    os.makedirs("results/deep", exist_ok=True)
    np.savez_compressed("cache/deep/baseline_scores.npz", **saved)
    json.dump(summary, open("results/deep/baseline_check.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
