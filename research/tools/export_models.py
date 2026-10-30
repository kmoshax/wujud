"""Export every HistGradientBoosting model in research/models/ to the JSON format read by crates/model
(format documented in crates/model/README.md, "Model JSON format").

    cd research && .venv/bin/python tools/export_models.py [--out ../crates/model/models] [--also cache/x.joblib ...]
"""
import argparse, glob, json, os
import sys
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
FORMAT = "hallsense-gbdt-v2"


def tree_json(predictor):
    """split nodes and leaves in separate arrays; a child index c >= 0 is a split node, c < 0 is leaf -1 - c.
    A +inf threshold (sklearn: "all non-missing values go left") is written as null."""
    nodes = predictor.nodes
    if nodes["is_categorical"].any():
        raise ValueError("categorical splits are not supported by the Rust reader")
    leaf = nodes["is_leaf"].astype(bool)
    split_ids, leaf_ids = np.nonzero(~leaf)[0], np.nonzero(leaf)[0]
    compact_id = np.zeros(len(nodes), int)
    compact_id[split_ids] = np.arange(len(split_ids))
    compact_id[leaf_ids] = -1 - np.arange(len(leaf_ids))
    splits = nodes[split_ids]
    if np.isnan(splits["num_threshold"]).any() or np.isneginf(splits["num_threshold"]).any():
        raise ValueError("thresholds must be finite or +inf (written as null)")
    return dict(
        feature=splits["feature_idx"].astype(int).tolist(),
        threshold=[None if np.isposinf(v) else float(v) for v in splits["num_threshold"]],
        missing_left=splits["missing_go_to_left"].astype(int).tolist(),
        left=compact_id[splits["left"]].astype(int).tolist(),
        right=compact_id[splits["right"]].astype(int).tolist(),
        leaf_value=nodes[leaf_ids]["value"].astype(float).tolist(),
    )


def loss_name(m):
    return {"HalfMultinomialLoss": "multinomial", "HalfBinomialLoss": "binomial",
            "HalfSquaredError": "squared_error"}[type(m._loss).__name__]


def model_json(m, source):
    if not isinstance(m, (HistGradientBoostingClassifier, HistGradientBoostingRegressor)):
        raise TypeError(type(m).__name__)
    if getattr(m, "_preprocessor", None) is not None or m.is_categorical_ is not None:
        raise ValueError("categorical features are not supported by the Rust reader")
    out = dict(format=FORMAT, source=source, loss=loss_name(m), n_features=int(m.n_features_in_),
               classes=[int(c) for c in m.classes_] if hasattr(m, "classes_") else None,
               baseline=[float(v) for v in np.ravel(m._baseline_prediction)],
               trees_per_iteration=int(m.n_trees_per_iteration_),
               iterations=[[tree_json(p) for p in it] for it in m._predictors])
    return out


def export(m, source, path):
    with open(path, "w") as f:
        json.dump(m, f, separators=(",", ":"))
    print(f"{source} -> {os.path.relpath(path, HERE)} ({os.path.getsize(path) / 1e6:.1f} MB)")


def export_file(src, out_dir):
    m = joblib.load(src)
    name = os.path.splitext(os.path.basename(src))[0]
    rel = os.path.relpath(src, HERE)
    if isinstance(m, dict) and "model" in m:                         # bench caches: {"model": ..., "cols": ...}
        m = m["model"]
    if hasattr(m, "m1") and hasattr(m, "mc"):                        # hallsense.model.SeatModel
        if m.feat_mask is not None:
            raise ValueError("SeatModel with a feature mask is not supported")
        seat = dict(format="hallsense-seat-model-v1", source=rel, stage1=model_json(m.m1, rel + "::m1"),
                    stage2=model_json(m.m2, rel + "::m2"), count=model_json(m.mc, rel + "::mc"))
        export(seat, rel, os.path.join(out_dir, name + ".json"))
        return
    export(model_json(m, rel), rel, os.path.join(out_dir, name + ".json"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "crates", "model", "models"))
    ap.add_argument("--also", nargs="*", default=[], help="extra joblib files (e.g. bench caches)")
    ap.add_argument("--only", nargs="*", help="export only these files instead of research/models/*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    srcs = a.only if a.only else sorted(glob.glob(os.path.join(HERE, "models", "*.joblib")))
    for src in srcs + a.also:
        export_file(src, a.out)
