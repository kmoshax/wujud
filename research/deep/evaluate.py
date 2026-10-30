"""Scores on real windows and the selection/holdout metrics (same definitions as bench/phase_exp.py::metrics)."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from deep.data import gather, model_input
from deep.preprocess import WINDOW

FIRST_MINUTE_WINDOWS = 9


@torch.no_grad()
def score_recordings(model, real, recordings, inputs, device, batch=256):
    """-> {recording: dict(probabilities [W,3], still [W])}"""
    model.eval()
    out = {}
    streams = (real.amp, real.phase, real.rssi)
    for r in recordings:
        starts = real.window_starts(r)
        probs, still = [], []
        for i in range(0, len(starts), batch):
            amp, phase, rssi = (x.to(device) for x in gather(streams, starts[i:i + batch], WINDOW))
            with torch.autocast("cuda", dtype=torch.bfloat16):
                _, logits, still_logit = model(model_input(amp, phase, rssi, inputs))
            probs.append(torch.softmax(logits.float(), 1).cpu().numpy()); still.append(still_logit.float().cpu().numpy())
        out[r] = dict(probabilities=np.concatenate(probs), still=np.concatenate(still))
    return out


def metrics(scores, labels):
    """scores {rec: {probabilities, still}}, labels {rec: 0/1/2}"""
    recs = list(scores)
    y = np.concatenate([[labels[r]] * len(scores[r]["still"]) for r in recs])
    prob = np.concatenate([scores[r]["probabilities"] for r in recs])
    still = np.concatenate([scores[r]["still"] for r in recs])
    predicted = prob.argmax(1)
    quiet = y < 2
    quiet_recs = [r for r in recs if labels[r] < 2]
    rec_y = [labels[r] for r in quiet_recs]
    first_minute = [scores[r]["still"][:FIRST_MINUTE_WINDOWS].mean() for r in quiet_recs]
    whole = [scores[r]["still"].mean() for r in quiet_recs]
    return dict(motion_recall=float((predicted[y == 2] == 2).mean()),
                motion_false_alarm=float((predicted[quiet] == 2).mean()),
                still_vs_empty_auc=float(roc_auc_score(y[quiet], still[quiet])),
                still_vs_empty_auc_recording=float(roc_auc_score(rec_y, first_minute)),
                still_vs_empty_auc_recording_all_windows=float(roc_auc_score(rec_y, whole)),
                n_windows=int(len(y)), n_recordings=len(recs))


def window_auc(scores, labels, recs):
    y = np.concatenate([[labels[r]] * len(scores[r]) for r in recs])
    s = np.concatenate([scores[r] for r in recs])
    return roc_auc_score(y, s)


def recording_auc(scores, labels, recs):
    return roc_auc_score([labels[r] for r in recs], [scores[r][:FIRST_MINUTE_WINDOWS].mean() for r in recs])


def bootstrap(still_scores, labels, reference_scores=None, reps=2000, seed=0):
    """95 % CIs of window and recording AUC (still vs empty), resampling recordings within each class;
    with reference_scores also the CI of the paired difference (this - reference)."""
    rng = np.random.default_rng(seed)
    empty = [r for r in still_scores if labels[r] == 0]
    still = [r for r in still_scores if labels[r] == 1]
    draws = {"window": [], "recording": [], "window_diff": [], "recording_diff": []}
    for _ in range(reps):
        recs = list(rng.choice(empty, len(empty))) + list(rng.choice(still, len(still)))
        draws["window"].append(window_auc(still_scores, labels, recs))
        draws["recording"].append(recording_auc(still_scores, labels, recs))
        if reference_scores is not None:
            draws["window_diff"].append(draws["window"][-1] - window_auc(reference_scores, labels, recs))
            draws["recording_diff"].append(draws["recording"][-1] - recording_auc(reference_scores, labels, recs))
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in draws.items() if v}


def per_device(still_scores, labels, meta):
    """window AUC inside each device that has both empty and still recordings in the split"""
    out = {}
    devices = {meta[r]["device"] for r in still_scores}
    for d in sorted(devices):
        recs = [r for r in still_scores if meta[r]["device"] == d and labels[r] < 2]
        if len({labels[r] for r in recs}) == 2:
            out[d] = dict(auc=float(window_auc(still_scores, labels, recs)),
                          empty=sum(labels[r] == 0 for r in recs), still=sum(labels[r] == 1 for r in recs))
    return out
