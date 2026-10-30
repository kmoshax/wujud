"""Chooses the variant on SELECTION only, then scores the saved models once on HOLDOUT and writes the comparison.

.venv/bin/python -m deep.report            -> results/deep/summary.json (+ chosen weights and ONNX if it beats the baseline)
"""
import glob, json, os, shutil
import numpy as np
import torch
from deep.data import RealRecordings, gather, noise_floor_db
from deep.evaluate import bootstrap, metrics, per_device, score_recordings
from deep.models import Detector

DEVICE = "cuda"
MAX_SELECTION_FALSE_ALARM = 0.02
BASELINE = "baseline (hand features + GBDT, sim only)"
NOISE_LEVEL = "reference: receiver noise level only (no training)"


def stored_scores(path, split, real, recs):
    z = np.load(path)
    return {r: dict(probabilities=z[f"{split}/{real.meta[r]['file']}/probabilities"],
                    still=z[f"{split}/{real.meta[r]['file']}/still"]) for r in recs}


def noise_level_scores(real, recs):
    """a still score that is just the window's receiver jitter: what a detector keyed on noise would output"""
    out = {}
    for r in recs:
        amp, _, _ = gather((real.amp, real.phase, real.rssi), real.window_starts(r), 400)
        level = noise_floor_db(amp)[:, 0, 0].numpy()
        out[r] = dict(probabilities=np.tile([0.5, 0.5, 0.0], (len(level), 1)), still=level)
    return out


def load_model(name, config):
    model = Detector(config["encoder"], len(config["inputs"])).to(DEVICE)
    model.load_state_dict(torch.load(f"cache/deep/runs/{name}.pt", map_location=DEVICE))
    return model


def evaluated(scores, labels, real, reference):
    quiet = {r: s["still"] for r, s in scores.items() if labels[r] < 2}
    out = metrics(scores, labels)
    out["ci"] = bootstrap(quiet, labels, reference)
    out["per_device"] = per_device(quiet, labels, real.meta)
    return out


def seed_groups(runs):
    """X, X_s1, X_s2 ... -> {X_ens<n>: [members]} when at least three seeds exist"""
    groups = {}
    for name in runs:
        groups.setdefault(name.rsplit("_s", 1)[0] if "_s" in name and name.rsplit("_s", 1)[1].isdigit() else name, []).append(name)
    return {f"{base}_ens{len(m)}": m for base, m in groups.items() if len(m) >= 3}


def averaged(score_sets):
    return {r: {k: np.mean([s[r][k] for s in score_sets], 0) for k in ("probabilities", "still")} for r in score_sets[0]}


def run_scores(name, run, split, real, recs):
    if split == "selection":
        return stored_scores(f"cache/deep/runs/{name}.scores.npz", split, real, recs)
    scores = score_recordings(load_model(name, run["config"]), real, recs, tuple(run["config"]["inputs"]), DEVICE)
    np.savez_compressed(f"cache/deep/runs/{name}.holdout.npz",
                        **{f"{real.meta[r]['file']}/{k}": v for r in recs for k, v in scores[r].items()})
    return scores


def chosen_on_selection(summary, candidates):
    eligible = [k for k in candidates if summary["variants"][k]["selection"]["motion_false_alarm"] <= MAX_SELECTION_FALSE_ALARM]
    return max(eligible, key=lambda k: summary["variants"][k]["selection"]["still_vs_empty_auc"])


def main():
    real = RealRecordings()
    runs = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob("results/deep/runs/*.json"))}
    ensembles = seed_groups(runs)
    summary = {"rule": "max selection window AUC with selection motion false alarm <= 2 %", "ensembles": ensembles, "variants": {}}
    for split in ("selection", "holdout"):
        recs = real.recordings(split)
        labels = {r: real.meta[r]["label"] for r in recs}
        base = stored_scores("cache/deep/baseline_scores.npz", split, real, recs)
        base_quiet = {r: s["still"] for r, s in base.items() if labels[r] < 2}
        summary["variants"].setdefault(BASELINE, {})[split] = evaluated(base, labels, real, None)
        summary["variants"].setdefault(NOISE_LEVEL, {})[split] = evaluated(noise_level_scores(real, recs), labels, real, base_quiet)
        split_scores = {}
        for name, run in runs.items():
            split_scores[name] = run_scores(name, run, split, real, recs)
            entry = summary["variants"].setdefault(name, {"config": run["config"], "sim_validation": run["sim_validation"]})
            entry[split] = evaluated(split_scores[name], labels, real, base_quiet)
        for name, members in ensembles.items():
            entry = summary["variants"].setdefault(name, {"config": runs[members[0]]["config"], "members": members})
            entry[split] = evaluated(averaged([split_scores[m] for m in members]), labels, real, base_quiet)
        if split == "selection":
            summary["chosen_on_selection"] = chosen_on_selection(summary, list(runs) + list(ensembles))
    json.dump(summary, open("results/deep/summary.json", "w"), indent=1)
    for name, v in summary["variants"].items():
        s, h = v["selection"], v["holdout"]
        print(f"{name:42s} sel {s['still_vs_empty_auc']:.3f} [{s['ci']['window'][0]:.2f},{s['ci']['window'][1]:.2f}] "
              f"rec {s['still_vs_empty_auc_recording']:.3f} | hold {h['still_vs_empty_auc']:.3f} "
              f"[{h['ci']['window'][0]:.2f},{h['ci']['window'][1]:.2f}] rec {h['still_vs_empty_auc_recording']:.3f} "
              f"motion {h['motion_recall']:.2f}/{h['motion_false_alarm']:.3f}")
    chosen = summary["chosen_on_selection"]
    print("chosen on selection:", chosen)
    if summary["variants"][chosen]["holdout"]["ci"].get("window_diff", [0])[0] > 0:
        for member in ensembles.get(chosen, [chosen]):
            export_winner(member, runs[member]["config"])


def export_winner(name, config):
    model = load_model(name, config).eval().float()
    shutil.copy(f"cache/deep/runs/{name}.pt", f"results/deep/{name}.pt")
    example = torch.zeros(1, len(config["inputs"]), 400, 52, device=DEVICE)
    torch.onnx.export(model, example, f"results/deep/{name}.onnx", input_names=["window"],
                      output_names=["features", "class_logits", "still_logit"], opset_version=17, dynamo=False)


if __name__ == "__main__":
    main()
