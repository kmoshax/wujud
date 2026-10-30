"""Test the ESP32 room path without an ESP32: replay REAL ESPectre recordings as the serial stream
firmware/hallsense_room prints (I,... banner, then P,<rssi>,<rx timestamp us>,<128 hex = 64 x [imag, real] int8>),
on pseudo-terminals, so live.py opens them through pyserial exactly like a board on USB.

One recording per fake board. Prints the pty paths and writes a room.json keyed by them:
    python tools/replay_espectre.py ~/Projects/ext/espectre/data/motion/<file>.npz --room-out cache/replay_room.json
    python live.py room --room cache/replay_room.json --ports /dev/pts/N
Real-time pacing from the recording's own timestamps; --speed 4 plays 4x faster, --loop repeats.
"""
import argparse, json, os, sys, threading, time, tty
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def lines(path, wrap_at=None):
    """recording -> [(seconds since start, serial line)] in the firmware's exact format.
    wrap_at=s shifts the device clock so its 32-bit us counter rolls over s seconds in (tests live.py's unwrap)."""
    z = np.load(path)
    ts = z["wifi_rx_ts_us"].astype(np.int64) & 0xFFFFFFFF            # rx_ctrl.timestamp is a 32-bit us counter
    t = np.unwrap(ts.astype(np.float64), period=2 ** 32) / 1e6
    if wrap_at is not None:
        ts = ((t - t.min()) * 1e6).astype(np.int64) + 2 ** 32 - int(wrap_at * 1e6)
        ts &= 0xFFFFFFFF
    raw = z["csi_data"].astype(np.int8)
    rssi = z["rssi_dbm"].astype(int)
    out = []
    for i in np.argsort(t, kind="stable"):
        if raw.shape[1] < 128:                                      # csi_cb drops info->len < 128
            continue
        out.append((t[i] - t.min(), f"P,{rssi[i]},{ts[i]},{raw[i, :128].view(np.uint8).tobytes().hex()}\n"))
    return out


def play(master, path, speed, loop, wrap_at=None):
    os.write(master, b"\nI,hallsense-room,ch=6,gw=192.168.1.1,mac=replay\n")
    while True:
        L = lines(path, wrap_at); t0 = time.time()
        for t, line in L:
            d = t0 + t / speed - time.time()
            if d > 0:
                time.sleep(d)
            os.write(master, line.encode())
        print(f"done: {os.path.basename(path)} ({len(L)} packets, {L[-1][0]:.0f} s)", flush=True)
        if not loop:
            return


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="ESPectre .npz recordings, one per fake board")
    ap.add_argument("--speed", type=float, default=1.0); ap.add_argument("--loop", action="store_true")
    ap.add_argument("--wrap-at", type=float, help="make the board's 32-bit us clock roll over this many seconds in")
    ap.add_argument("--room-out", help="write a room.json (geometry of room.example.json) keyed by the pty paths")
    a = ap.parse_args()
    ptys = []
    for f in a.files:
        m, s = os.openpty(); tty.setraw(s)                          # raw: no echo, no CR/LF rewriting
        ptys.append((m, s, os.ttyname(s), f))
    names = [p[2] for p in ptys]
    if a.room_out:
        r = json.load(open(os.path.join(HERE, "room.example.json")))
        spots = list(r["boards"].values())
        r["boards"] = {n: spots[k % len(spots)] for k, n in enumerate(names)}
        json.dump(r, open(a.room_out, "w"), indent=1)
    print("ports:", " ".join(names), flush=True)
    th = [threading.Thread(target=play, args=(m, f, a.speed, a.loop, a.wrap_at), daemon=True) for m, s, n, f in ptys]
    for x in th:
        x.start()
    for x in th:
        x.join()
    time.sleep(2)                                                   # let the reader drain the last bytes


if __name__ == "__main__":
    main()
