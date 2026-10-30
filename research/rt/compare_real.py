"""Do ray-traced sessions look like the board's own recordings? The same few statistics for both.

    python compare_real.py <rt dir> <recording.csv> <lying start-end> <walking start-end>   (HH:MM:SS)
"""

import glob
import sys
import time

import numpy as np

HALF_SECOND_PACKETS = 50


def amplitudes_db(iq):
    """`iq` [packets, 64, 2] int8 -> dB of the signal bins, each packet's mean removed (AGC)."""
    a = np.hypot(iq[..., 0].astype(float), iq[..., 1].astype(float))
    signal = a.mean(0) > 2
    db = 20 * np.log10(a[:, signal] + 1)
    return db - db.mean(1, keepdims=True)


def stats(iq):
    db = amplitudes_db(iq)
    step = np.median(np.sqrt(((db[1:] - db[:-1]) ** 2).mean(1)))
    chunks = [db[i : i + HALF_SECOND_PACKETS] for i in range(0, len(db) - HALF_SECOND_PACKETS, HALF_SECOND_PACKETS)]
    disturbance = np.median([c.std(0).mean() for c in chunks])
    spread = db.mean(0).std()
    return step, disturbance, spread


def recording_packets(path, start, end):
    day = time.strftime("%Y-%m-%d", time.localtime(float(open(path).readline().split(",")[0])))
    t0, t1 = (time.mktime(time.strptime(f"{day} {s}", "%Y-%m-%d %H:%M:%S")) for s in (start, end))
    rows = []
    for line in open(path):
        fields = line.rstrip().split(",")
        if len(fields) >= 6 and fields[2] == "P" and t0 <= float(fields[0]) <= t1:
            rows.append(np.frombuffer(bytes.fromhex(fields[5]), np.int8).reshape(64, 2))
    return np.array(rows)


def main():
    rt_dir, recording, lying, walking = sys.argv[1:5]
    print(f"{'':28}{'step dB':>9}{'0.5 s disturbance dB':>22}{'profile spread dB':>19}")
    for activity in ("away", "lie", "sit", "stand", "walk"):
        rows = [stats(np.load(f)["iq"][k]) for f in glob.glob(f"{rt_dir}/*_{activity}.npz") for k in range(np.load(f)["iq"].shape[0])]
        m = np.median(rows, 0)
        print(f"ray-traced {activity:17}{m[0]:9.2f}{m[1]:22.2f}{m[2]:19.2f}")
    for name, span in (("lying", lying), ("walking", walking)):
        s = stats(recording_packets(recording, *span.split("-")))
        print(f"real {name:23}{s[0]:9.2f}{s[1]:22.2f}{s[2]:19.2f}")


if __name__ == "__main__":
    main()
