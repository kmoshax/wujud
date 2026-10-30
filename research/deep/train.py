"""Train one sim-to-real variant and score it on the real selection and holdout splits.

.venv/bin/python -m deep.train --name cnn2d_plain --encoder cnn2d [--augment 1] [--warp] [--dann 0.3] [--ssl 2000]
Writes results/deep/runs/<name>.json (metrics) and cache/deep/runs/<name>.{pt,scores.npz}.
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from deep.augment import augment, time_warp
from deep.data import SimSessions, RealRecordings, gather, gather_sessions, model_input
from deep.evaluate import metrics, per_device, score_recordings
from deep.models import Detector
from deep.preprocess import WINDOW

SESSION_GRID = 1200
DEVICE = "cuda"


def sim_batch(sim, rng, batch, warp):
    length = int(round(WINDOW * rng.uniform(0.8, 1.25))) if warp else WINDOW
    sessions = sim.balanced_sessions(rng, batch)
    starts = (rng.random(batch) * (SESSION_GRID - length)).astype(int)
    amp, phase, rssi = (x.to(DEVICE, non_blocking=True) for x in gather_sessions(sim, sessions, starts, length))
    return amp, phase, rssi, sim.label[sessions].to(DEVICE)


def real_batch(real, rng, batch, role="train"):
    starts = real.random_crops(rng, role, batch)
    return [x.to(DEVICE, non_blocking=True) for x in gather((real.amp, real.phase, real.rssi), starts, WINDOW)]


def prepared(amp, phase, rssi, args, augment_level):
    if augment_level > 0:
        amp, phase, rssi = augment(amp, phase, rssi, augment_level)
    if amp.shape[1] != WINDOW:
        amp, phase, rssi = time_warp(amp, phase, rssi)
    return model_input(amp, phase, rssi, args.inputs)


def nt_xent(a, b, temperature=0.2):
    z = torch.cat([a, b])
    similarity = z @ z.T / temperature
    similarity.fill_diagonal_(-1e9)
    n = len(a)
    target = torch.cat([torch.arange(n, 2 * n), torch.arange(n)]).to(z.device)
    return F.cross_entropy(similarity, target)


def contrastive_pretraining(model, sim, real, rng, args):
    """two nearby, independently augmented crops of the same stream are positives; half real (unlabeled), half sim"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    half = args.batch // 2
    for step in range(args.ssl):
        real_starts = real.random_crops(rng, "train", half, WINDOW + 50)
        sessions = rng.choice(sim.train_index, half)
        sim_starts = (rng.random(half) * (SESSION_GRID - WINDOW - 50)).astype(int)
        views = []
        for shift in (rng.integers(0, 51, half), rng.integers(0, 51, half)):
            r = gather((real.amp, real.phase, real.rssi), real_starts + shift, WINDOW)
            s = gather_sessions(sim, sessions, sim_starts + shift, WINDOW)
            amp, phase, rssi = (torch.cat([x, y]).to(DEVICE) for x, y in zip(r, s))
            views.append(prepared(amp, phase, rssi, args, 1.0))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            za, zb = model.embedding(model.encoder(views[0])), model.embedding(model.encoder(views[1]))
            loss = nt_xent(za.float(), zb.float())
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if step % 500 == 0:
            print(f"  ssl {step} loss {loss.item():.3f}", flush=True)


def supervised_training(model, sim, real, rng, args):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, total_steps=args.steps, pct_start=0.1)
    for step in range(args.steps):
        amp, phase, rssi, y = sim_batch(sim, rng, args.batch, args.warp)
        x = prepared(amp, phase, rssi, args, args.augment)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if args.dann > 0:
                xr = prepared(*real_batch(real, rng, args.batch), args, 0.0)
                z_all, logits_all, still_all = model(torch.cat([x, xr]))
                z, logits, still = z_all[:len(x)], logits_all[:len(x)], still_all[:len(x)]
                progress = step / args.steps
                weight = args.dann * (2 / (1 + np.exp(-10 * progress)) - 1)
                domain = model.domain_logit(z_all.float(), weight)
                domain_target = torch.cat([torch.zeros(len(x)), torch.ones(len(xr))]).to(DEVICE)
                domain_loss = F.binary_cross_entropy_with_logits(domain.float(), domain_target)
            else:
                z, logits, still = model(x)
                domain_loss = torch.zeros((), device=DEVICE)
            quiet = y < 2
            loss = F.cross_entropy(logits.float(), y, label_smoothing=0.05) \
                + F.binary_cross_entropy_with_logits(still.float()[quiet], y[quiet].float()) + domain_loss
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step(); schedule.step()
        if step % 500 == 0:
            print(f"  step {step} loss {loss.item():.3f} domain {domain_loss.item():.3f}", flush=True)


@torch.no_grad()
def sim_validation(model, sim, args, rng):
    model.eval()
    sessions = sim.validation_index
    starts = (rng.random(len(sessions)) * (SESSION_GRID - WINDOW)).astype(int)
    probs, still = [], []
    for i in range(0, len(sessions), 256):
        amp, phase, rssi = (x.to(DEVICE) for x in gather_sessions(sim, sessions[i:i + 256], starts[i:i + 256], WINDOW))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, logits, s = model(model_input(amp, phase, rssi, args.inputs))
        probs.append(logits.float().argmax(1).cpu()); still.append(s.float().cpu())
    y = sim.label[sessions].numpy(); predicted = torch.cat(probs).numpy(); s = torch.cat(still).numpy()
    return dict(accuracy=float((predicted == y).mean()), still_vs_empty_auc=float(roc_auc_score(y[y < 2], s[y < 2])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--encoder", default="cnn2d")
    p.add_argument("--inputs", default="amp")
    p.add_argument("--augment", type=float, default=0.0)
    p.add_argument("--warp", action="store_true")
    p.add_argument("--dann", type=float, default=0.0)
    p.add_argument("--ssl", type=int, default=0)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--holdout", action="store_true", help="also score the holdout split")
    args = p.parse_args()
    args.inputs = tuple(args.inputs.split("+"))
    torch.manual_seed(args.seed); rng = np.random.default_rng(args.seed)
    sim, real = SimSessions(), RealRecordings()
    model = Detector(args.encoder, len(args.inputs)).to(DEVICE)
    began = time.time()
    if args.ssl:
        contrastive_pretraining(model, sim, real, rng, args)
    supervised_training(model, sim, real, rng, args)
    result = dict(config={k: v for k, v in vars(args).items()}, sim_validation=sim_validation(model, sim, args, rng),
                  minutes=(time.time() - began) / 60, parameters=sum(p.numel() for p in model.parameters()))
    splits = ["selection"] + (["holdout"] if args.holdout else [])
    saved = {}
    for split in splits:
        recs = real.recordings(split)
        scores = score_recordings(model, real, recs, args.inputs, DEVICE)
        labels = {r: real.meta[r]["label"] for r in recs}
        result[split] = metrics(scores, labels)
        result[split]["per_device"] = per_device({r: scores[r]["still"] for r in recs if labels[r] < 2}, labels, real.meta)
        for r in recs:
            saved[f"{split}/{real.meta[r]['file']}/probabilities"] = scores[r]["probabilities"]
            saved[f"{split}/{real.meta[r]['file']}/still"] = scores[r]["still"]
        m = result[split]
        print(f"{args.name:28s} {split:9s} motion {m['motion_recall']:.3f}/{m['motion_false_alarm']:.3f} "
              f"AUC win {m['still_vs_empty_auc']:.3f} rec {m['still_vs_empty_auc_recording']:.3f} "
              f"(all {m['still_vs_empty_auc_recording_all_windows']:.3f})  sim-val {result['sim_validation']}", flush=True)
    os.makedirs("results/deep/runs", exist_ok=True); os.makedirs("cache/deep/runs", exist_ok=True)
    json.dump(result, open(f"results/deep/runs/{args.name}.json", "w"), indent=1)
    torch.save(model.state_dict(), f"cache/deep/runs/{args.name}.pt")
    np.savez_compressed(f"cache/deep/runs/{args.name}.scores.npz", **saved)


if __name__ == "__main__":
    main()
