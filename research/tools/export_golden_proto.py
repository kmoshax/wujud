"""Golden data for crates/proto: real serial lines and the values the Python host gets from them.

Room: P lines are produced by tools/replay_espectre.py::lines from a real ESPectre recording (with the
device clock shifted so its 32-bit counter wraps inside the excerpt), and decoded with live.py::parse_p
plus the timestamp unwrap of live.py's room reader.
Hall: there is no real mesh recording yet, so R lines and ESP-NOW report frames are made by emulating
firmware/wujud_node.ino (flush_reports + print_rec) on the same real CSI packets, then decoded the way
live.py::Windower.add_line does.

Run from research/:  .venv/bin/python tools/export_golden_proto.py
"""
import glob, json, os, struct, sys
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "tools"))
import replay_espectre
from live import parse_p
from wujud.config import N_GROUPS
from wujud.features import packet_groups

OUT = os.path.join(HERE, "..", "crates", "proto", "tests", "golden")
RECORDING = sorted(glob.glob(os.path.expanduser("~/Projects/ext/espectre/data/motion/motion_c3_*.npz")))[0]
N_ROOM_LINES = 300
WRAP_AT_S = 1.0
OUTLIER_DB = 8.0
EMA_ALPHA = 0.1
MIN_PACKETS = 3
RECEIVER, SOURCE = 0, 1
MAGIC, T_REPORT = 0x48, 2


def unwrap_like_live_reader(timestamps):
    out, t_prev, wraps = [], None, 0
    for ts in timestamps:
        if t_prev is not None and ts < t_prev:
            wraps += 1
        t_prev = ts
        out.append((ts + wraps * 2 ** 32) / 1e6)
    return out


def room_golden():
    lines = [line for _, line in replay_espectre.lines(RECORDING, wrap_at=WRAP_AT_S)[:N_ROOM_LINES]]
    parsed = [parse_p(line) for line in lines]
    seconds = unwrap_like_live_reader([ts for _, ts, _ in parsed])
    return dict(
        source=os.path.basename(RECORDING),
        lines=lines,
        expected=[dict(rssi=int(rssi), rx_timestamp_us=int(ts), csi=raw.reshape(-1).astype(int).tolist(), seconds=t)
                  for (rssi, ts, raw), t in zip(parsed, seconds)],
    )


def lltf_iq_like_node_ino(raw128):
    """wujud_node.ino::csi_cb indexing: k>0 -> bin k, k<0 -> bin 64+k, order -26..-1, 1..26."""
    bins = [64 + k if k < 0 else k for k in range(-26, 27) if k != 0]
    return raw128.reshape(64, 2)[bins]


def emulate_node_windows(raw, rssi, t):
    """wujud_node.ino: EMA RSSI gate (seeded with the first packet), 1 s windows, n >= 3, quantize."""
    groups = packet_groups(np.stack([lltf_iq_like_node_ino(r) for r in raw]), rssi.astype(np.float32)).astype(np.float64)
    ema, window_of = None, np.floor((t - t[0])).astype(int)
    records = []
    for w in range(window_of.max() + 1):
        acc = []
        for n in np.flatnonzero(window_of == w):
            if ema is None:
                ema = float(rssi[n])
            if abs(rssi[n] - ema) <= OUTLIER_DB:
                ema += EMA_ALPHA * (rssi[n] - ema)
                acc.append(groups[n])
        if len(acc) < MIN_PACKETS:
            continue
        acc = np.array(acc)
        mu = acc.mean(0)
        sd = np.sqrt(np.maximum((acc ** 2).mean(0) - mu ** 2, 0))
        records.append(dict(source=SOURCE, packets=min(len(acc), 255),
                            mean_q4=[int(np.sign(v) * np.floor(abs(v) * 4 + 0.5)) for v in mu],
                            std_q20=[255 if v * 20 > 255 else int(np.floor(v * 20 + 0.5)) for v in sd]))
    return records


def print_rec_like_node_ino(receiver, rec):
    means = ",".join("%.2f" % (np.float32(q) / np.float32(4.0)) for q in rec["mean_q4"])
    stds = ",".join("%.2f" % (np.float32(q) / np.float32(20.0)) for q in rec["std_q20"])
    return f"R,{receiver},{rec['source']},{rec['packets']},{means},{stds}"


def windower_values(line):
    p = line.strip().split(",")
    assert len(p) == 4 + 2 * N_GROUPS and p[0] == "R"
    v = np.array(p[4:], float)
    return dict(receiver=int(p[1]), source=int(p[2]), packets=int(p[3]),
                mean_db=v[:N_GROUPS].tolist(), std_db=v[N_GROUPS:].tolist())


def report_frame_bytes(receiver, seq, recs):
    out = struct.pack("<BBBBH", MAGIC, T_REPORT, receiver, len(recs), seq)
    for r in recs:
        out += struct.pack("<BB13h13B", r["source"], r["packets"], *r["mean_q4"], *r["std_q20"])
    return out


def hall_golden():
    z = np.load(RECORDING)
    raw = z["csi_data"].astype(np.int8)[:, :128]
    rssi = z["rssi_dbm"].astype(int)
    ts = np.unwrap((z["wifi_rx_ts_us"].astype(np.int64) & 0xFFFFFFFF).astype(np.float64), period=2 ** 32) / 1e6
    order = np.argsort(ts, kind="stable")
    records = emulate_node_windows(raw[order], rssi[order], ts[order])
    lines = [print_rec_like_node_ino(RECEIVER, r) for r in records]
    frames = [dict(seq=k, hex=report_frame_bytes(RECEIVER, k, records[i:i + 5]).hex(), records=records[i:i + 5])
              for k, i in enumerate(range(0, min(len(records), 15), 5))]
    return dict(source=os.path.basename(RECORDING), receiver=RECEIVER, records=records, lines=lines,
                expected=[windower_values(l) for l in lines], report_frames=frames)


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, data in (("room_lines.json", room_golden()), ("hall_reports.json", hall_golden())):
        path = os.path.join(OUT, name)
        json.dump(data, open(path, "w"))
        print(f"wrote {os.path.relpath(path, HERE)} ({os.path.getsize(path) // 1024} KiB)")


if __name__ == "__main__":
    main()
