"""The guided test for the room mesh: what to do now in big letters, a countdown and a bell at every
change, while every link the hub sends is recorded next to what the person was doing.

    cd research && .venv/bin/python tools/guided_room.py --hub 192.168.1.40
    open http://127.0.0.1:8771

Recorded to rec/guided_room_<time>.csv (the live app's --record format) and <same>_plan.json (the
steps with their unix start and end). The hub batches lines into datagrams; any number per datagram.
"""
import argparse
import json
import os
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from guided_test import STEPS

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARABIC = {
    "LEAVE the room and close the door": "اخرج من الأوضة واقفل الباب",
    "COME IN and WALK around": "ادخل وامشي في الأوضة",
    "SIT on a chair and keep still": "اقعد على كرسي وما تتحركش",
    "LIE on the bed and keep still": "نام على السرير وما تتحركش",
    "GET UP and WALK around": "قوم وامشي",
    "STAND still": "اقف ثابت",
}
PAGE = """<!doctype html><meta charset=utf-8><title>wujud guided test</title>
<style>body{background:#0e1118;color:#e8ecf3;font-family:sans-serif;text-align:center;margin:0}
#what{font-size:9vw;margin-top:18vh;direction:rtl}#left{font-size:14vw;color:#6ec8ff}
#next{font-size:3vw;color:#8a93a6;direction:rtl}button{font-size:4vw;padding:2vw 5vw;margin-top:10vh}</style>
<div id=what>جاهز؟</div><div id=left></div><div id=next></div>
<button id=go onclick="fetch('/start',{method:'POST'}).then(r=>r.json()).then(p=>{plan=p;go.remove()})">ابدأ</button>
<script>
let plan=null,shown=-1;
function bell(){const a=new AudioContext(),o=a.createOscillator();o.frequency.value=880;o.connect(a.destination);o.start();o.stop(a.currentTime+0.4)}
setInterval(()=>{if(!plan)return;const now=Date.now()/1000;
 let i=plan.findIndex(s=>now<s.end);
 if(now<plan[0].start){what.textContent='استعد: '+plan[0].text;left.textContent=Math.ceil(plan[0].start-now);next.textContent='';return}
 if(i<0){what.textContent='خلصنا، شكراً';left.textContent='';next.textContent='';return}
 if(i!=shown){shown=i;bell()}
 what.textContent=plan[i].text;left.textContent=Math.ceil(plan[i].end-now);
 next.textContent=plan[i+1]?('بعدها: '+plan[i+1].text):'';},250);
</script>"""


class Test:
    def __init__(self, hub, prep):
        self.hub, self.prep = hub, prep
        self.base = os.path.join(HERE, "rec", time.strftime("guided_room_%Y%m%d_%H%M"))
        self.out = open(self.base + ".csv", "w")
        self.lines = 0
        threading.Thread(target=self.listen, daemon=True).start()

    def start(self):
        t = time.time() + self.prep
        plan = []
        for label, seconds, text in STEPS:
            plan.append({"start": t, "end": t + seconds, "label": label, "text": ARABIC[text], "english": text})
            t += seconds
        json.dump(plan, open(self.base + "_plan.json", "w"), ensure_ascii=False, indent=1)
        print(f"plan: {len(plan)} steps, {t - time.time():.0f} s, recording to {self.base}.csv", flush=True)
        return plan

    def listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.2)
        asked = 0.0
        while True:
            now = time.time()
            try:
                if now - asked > 1.0:
                    asked = now
                    sock.sendto(b"wujud?", (self.hub, 5005))
                data, _ = sock.recvfrom(4096)
            except OSError:
                continue
            for line in data.decode(errors="ignore").splitlines():
                if "|P," in line:
                    device, rest = line.split("|", 1)
                    self.out.write(f"{now:.3f},udp:{self.hub}@{device},{rest}\n")
                    self.lines += 1
            self.out.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", default="192.168.1.40")
    parser.add_argument("--prep", type=int, default=15, help="seconds between pressing start and the first step")
    parser.add_argument("--port", type=int, default=8771)
    args = parser.parse_args()
    test = Test(args.hub, args.prep)

    class Page(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send(self, body, kind):
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(body.encode())

        def do_GET(self):
            self.send(PAGE, "text/html; charset=utf-8")

        def do_POST(self):
            self.send(json.dumps(test.start(), ensure_ascii=False), "application/json")

    print(f"open http://127.0.0.1:{args.port}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Page).serve_forever()


if __name__ == "__main__":
    main()
