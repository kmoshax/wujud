#!/bin/bash
# The presentation in one line each:
#   research/present.sh room     the simulated room in 3D, full screen (no boards needed)
#   research/present.sh hall     the simulated lecture hall counting students
#   research/present.sh replay   a real recording of the room played back in 3D
#   research/present.sh live     the real boards (the hub on its own Wi-Fi "wujud" or the home network)
#   research/present.sh panel    the simulated room with the numbers panel (egui)
# Keys in the 3D view: f full screen, 1/2/3 room views, 4 follow the person, space slow turn, q quit.
set -euo pipefail
cd "$(dirname "$0")/.."

replay_file=research/rec/guided_room_20261101_2120_test.csv
hub_addresses=(192.168.4.1 192.168.1.42 192.168.1.40)

find_hub() {
    for address in "${hub_addresses[@]}"; do
        ping -c 1 -W 1 "$address" >/dev/null 2>&1 && { echo "udp:$address"; return; }
    done
    echo "no hub answered at ${hub_addresses[*]}; set BOARD=udp:<address>" >&2
    exit 1
}

wujud() {
    cargo build --release -q -p wujud
    exec target/release/wujud "$@"
}

case "${1:-}" in
    room) wujud view-demo --boards "${BOARDS:-4}" --fullscreen ;;
    hall) wujud demo-hall ;;
    replay)
        [[ -f $replay_file ]] || { echo "missing $replay_file" >&2; exit 1; }
        wujud view --ports "replay:$replay_file" --fullscreen ;;
    live) wujud view --ports "${BOARD:-$(find_hub)}" --fullscreen ;;
    panel) wujud demo --boards "${BOARDS:-3}" ;;
    *)
        sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'
        exit 1 ;;
esac
