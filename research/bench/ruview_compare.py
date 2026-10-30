"""Head-to-head on REAL ESP32 recordings: RuView's on-device detector vs wujud's simulation-trained model.

RuView detector = a line-by-line port of firmware/esp32-csi-node/main/edge_processing.c (process_frame, steps 1-9):
frames gated to EDGE_DSP_SAMPLE_HZ (20 Hz; 8 Hz on C6) as in csi_collector.c ->
raw per-subcarrier phase atan2(byte1, byte0) -> unwrap in time -> running (Welford) variance per subcarrier ->
top-1 subcarrier re-picked every 100 frames -> motion_energy = variance of the last 20 primary phases ->
presence = motion_energy > adaptive threshold (mean + 3 sigma of the first 1200 frames), Schmitt hysteresis 0.5,
clear after 5 frames; motion flag = motion_energy > 0.01 (send_vitals_packet).

Both systems are scored on the same ESPectre test split (selection + holdout) and the same 20 s / 5 s windows.
wujud never saw a real sample (models/room_sim_all.joblib was trained on simulation only).

RuView side runs RuView's own compiled C by default (bench/ruview_native/build.sh first);
RUVIEW_IMPL=python uses the line-by-line Python port instead (kept to cross-check it).
ESPECTRE_DATA=/path/to/espectre/data python bench/ruview_compare.py
"""
import json, os, sys
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from wujud.roomfeat import windows, amplitude_from_iq, WIN_S, HOP_S

ROOT = os.environ.get("ESPECTRE_DATA", os.path.expanduser("~/Projects/ext/espectre/data"))
LAB = {"empty": 0, "static_presence": 1, "motion": 2}
CALIB_FRAMES, SIGMA_MULT, HYST, CLEAR = 1200, 3.0, 0.5, 5        # edge_processing.h
MAX_SC, TOPK_EVERY, ME_WIN = 128, 100, 20
EDGE_HZ = {"C6": 8}                                                         # Kconfig EDGE_DSP_SAMPLE_HZ, default 20


def edge_gate(t, hz):
    """csi_collector.c: frames enter the edge DSP on a fixed clock (deadline advanced by whole periods)"""
    I, nxt, keep = 1.0 / hz, t[0], []
    for i, ti in enumerate(t):
        if ti >= nxt:
            keep.append(i); nxt += (np.floor((ti - nxt) / I) + 1) * I
    return np.array(keep)


def ruview_edge(iq_bytes):
    """iq_bytes [n, 2K] int8 in the ESP32 buffer order -> per-frame motion_energy, presence flag, threshold"""
    n = len(iq_bytes)
    K = min(iq_bytes.shape[1] // 2, MAX_SC)
    b = iq_bytes[:, :2 * K].astype(np.float64)
    ph = np.unwrap(np.arctan2(b[:, 1::2], b[:, 0::2]), axis=0)              # steps 1-2 (same as the per-frame unwrap)
    # step 3: Welford variance up to and including frame i (sample variance, count-1)
    c1, c2 = np.cumsum(ph, 0), np.cumsum(ph ** 2, 0)
    cnt = np.arange(1, n + 1)[:, None]
    var = np.where(cnt > 1, (c2 - c1 ** 2 / cnt) / np.maximum(cnt - 1, 1), 0.0)
    # step 4-5: primary subcarrier, re-picked at frame_count % 100 == 1 (frame_count starts at 1)
    prim = np.zeros(n, int); cur = 0
    for i in range(n):
        if (i + 1) % TOPK_EVERY == 1:
            cur = int(np.argmax(var[i]))                                    # first max wins, like the C loop
        prim[i] = cur
    hist = ph[np.arange(n), prim]
    # step 8: variance of the last min(len,20) primary phases, once len >= 10
    me = np.zeros(n)
    for i in range(9, n):
        w = hist[max(0, i - ME_WIN + 1):i + 1]
        me[i] = max(0.0, np.mean(w ** 2) - np.mean(w) ** 2)
    # step 9: adaptive calibration over the first 1200 motion values, then Schmitt + clear debounce
    cal = me[:CALIB_FRAMES]
    thr = max(0.01, cal.mean() + SIGMA_MULT * cal.std()) if n >= CALIB_FRAMES else None
    pres = np.zeros(n, bool); prev = False; below = 0
    for i in range(n):
        t = thr if (thr is not None and i >= CALIB_FRAMES) else 0.05
        s = me[i]
        if s > t:
            below = 0; prev = True
        elif s >= t * HYST:
            below = 0
        else:
            below += 1
            if prev and below >= CLEAR:
                below = 0; prev = False
        pres[i] = prev
    return me, pres, thr


NATIVE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ruview_native", "replay")


def ruview_native(iq_bytes, t, rssi, channel):
    """the same frames through RuView's own compiled edge_processing.c (bench/ruview_native) -> me, presence, threshold"""
    import struct, subprocess
    buf = bytearray()
    for i in range(len(iq_bytes)):
        b = iq_bytes[i].astype(np.int8).tobytes()
        buf += struct.pack("<QbBH", int(t[i] * 1e6) & 0xFFFFFFFFFFFFFFFF, int(rssi[i]), int(channel), len(b)) + b
    out = subprocess.run([NATIVE], input=bytes(buf), capture_output=True, check=True).stdout.decode().split("\n")
    a = np.array([[float(x) for x in l.split()] for l in out if l.strip()])
    thr = float(a[-1, 3]) if a[-1, 2] else None
    return a[:, 0], a[:, 1].astype(bool), thr


def window_index(t, n_windows):
    """frame -> window membership, aligned with roomfeat.windows (grid starts at t[0])"""
    return [(t >= t[0] + k * HOP_S) & (t < t[0] + k * HOP_S + WIN_S) for k in range(n_windows)]


def main():
    info = json.load(open(os.path.join(ROOT, "dataset_info.json")))
    model = joblib.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "room_sim_all.joblib"))
    rows = []
    for lab, files in info["files"].items():
        for m in files:
            if m.get("dataset_role") not in ("selection", "holdout"):
                continue
            z = np.load(os.path.join(ROOT, lab, m["filename"]))
            t = np.unwrap(z["wifi_rx_ts_us"].astype(np.float64), period=2 ** 32) / 1e6
            o = np.argsort(t); t = t[o]; raw = z["csi_data"][o]
            X, _ = windows(t, amplitude_from_iq(raw.reshape(len(raw), -1, 2)), z["rssi_dbm"][o].astype(float))
            g = edge_gate(t, EDGE_HZ.get(m["chip"], 20))
            if os.environ.get("RUVIEW_IMPL", "native") == "native":
                ch = int(np.atleast_1d(z["channel"])[0]) if "channel" in z else 0
                me_g, pres_g, thr = ruview_native(raw[g], t[g] - t[0], z["rssi_dbm"][o][g], ch)
            else:
                me_g, pres_g, thr = ruview_edge(raw[g])
            me = np.full(len(t), np.nan); pres = np.zeros(len(t), bool)
            me[g] = me_g; pres[g] = pres_g; gate = np.zeros(len(t), bool); gate[g] = True
            for k, sel in enumerate(window_index(t, len(X))):
                sel = sel & gate
                if sel.sum() < 10:
                    continue
                pr = model.predict_proba(X[k:k + 1])[0]
                rows.append(dict(file=m["filename"], chip=m["chip"], y=LAB[lab], k=k,
                                 hs_pred=int(np.argmax(pr)), hs_still=float(pr[1] / (pr[0] + pr[1] + 1e-9)), hs_move=float(pr[2]),
                                 rv_me=float(np.mean(me[sel])), rv_move=bool(np.mean(me[sel] > 0.01) > 0.5),
                                 rv_pres=bool(np.mean(pres[sel]) > 0.5), rv_calibrated=thr is not None and bool(np.any(np.searchsorted(g, np.where(sel)[0]) >= CALIB_FRAMES))))
            print(f"{lab:16s} {m['chip']:6s} windows={len(X):3d} thr={thr if thr is None else round(thr, 4)}", flush=True)
    y = np.array([r["y"] for r in rows])
    res = {"n_windows": len(rows), "n_recordings": len({r["file"] for r in rows})}
    for name, move, pmove in (("wujud (sim-only)", np.array([r["hs_pred"] == 2 for r in rows]), np.array([r["hs_move"] for r in rows])),
                              ("RuView edge", np.array([r["rv_move"] for r in rows]), np.array([r["rv_me"] for r in rows]))):
        res[name] = dict(motion_recall=float(move[y == 2].mean()), motion_false_alarm=float(move[y < 2].mean()),
                         motion_auc=float(roc_auc_score(y == 2, pmove)))
    s = y < 2
    res["wujud (sim-only)"]["still_vs_empty_auc"] = float(roc_auc_score(y[s], np.array([r["hs_still"] for r in rows])[s]))
    res["RuView edge"]["still_vs_empty_auc"] = float(roc_auc_score(y[s], np.array([r["rv_me"] for r in rows])[s]))
    # RuView's own yes/no presence flag, only in windows after its 1200-frame boot calibration
    c = np.array([r["rv_calibrated"] for r in rows]) & s
    p = np.array([r["rv_pres"] for r in rows])
    res["RuView edge"]["presence_flag_after_calibration"] = dict(windows=int(c.sum()), says_present_when_empty=float(p[c & (y == 0)].mean()),
                                                               says_present_when_still=float(p[c & (y == 1)].mean()))
    hp = np.array([r["hs_pred"] for r in rows])
    res["wujud (sim-only)"]["presence_call"] = dict(says_present_when_empty=float((hp[y == 0] >= 1).mean()), says_present_when_still=float((hp[y == 1] >= 1).mean()))
    os.makedirs("results", exist_ok=True)
    res["ruview_impl"] = os.environ.get("RUVIEW_IMPL", "native")
    json.dump(res, open(f"results/ruview_compare_{res['ruview_impl']}.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
