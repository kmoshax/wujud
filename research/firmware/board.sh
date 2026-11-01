#!/usr/bin/env bash
# The room boards, from the laptop. Over USB when one is plugged in, otherwise over Wi-Fi
# (the laptop joined to the hub's network "wujud").
#   board.sh flash                     build research/firmware/wujud_room and send it; over Wi-Fi it goes
#                                      to every node the hub lists, then to the hub
#                                      (OVER_WIFI=1 sends it over Wi-Fi even when plugged in)
#   board.sh status                    what the hub and every node are doing
#   board.sh set ping_hz=200 channel=6 change the hub's settings (the hub reboots)
#   board.sh reboot                    reboot the hub
# BOARD=192.168.4.x talks to that board instead of the hub.
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
sketch="$here/wujud_room"
key_file="$sketch/ota_key.h"
board="http://${BOARD:-192.168.4.1}"

if [[ ! -f "$key_file" ]]; then
  echo "#define OTA_KEY \"$(head -c 18 /dev/urandom | base64 | tr '+/' 'xy')\"" > "$key_file"
fi
key="$(sed -n 's/.*OTA_KEY "\(.*\)".*/\1/p' "$key_file")"
call() { curl --fail --silent --show-error --max-time 20 --user "wujud:$key" "$@"; }

flash() {
  local build
  build="$(mktemp -d)"
  trap 'rm -rf "$build"' RETURN
  arduino-cli compile --fqbn esp32:esp32:esp32 --output-dir "$build" "$sketch"
  local port
  port="$(ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null | head -1 || true)"
  [[ -n "${OVER_WIFI:-}" ]] && port=""
  if [[ -n "$port" ]]; then
    arduino-cli upload --fqbn esp32:esp32:esp32 --port "$port" --input-dir "$build" \
      --board-options UploadSpeed=921600 "$sketch"
  else
    local target
    for target in $(node_ips) "${BOARD:-192.168.4.1}"; do
      # The update page refuses uploads whose Origin is not the board itself (CSRF guard).
      # A megabyte to a board that is also busy measuring: give it longer than a status request.
      call --max-time 180 -H "Origin: http://$target" -F "firmware=@$build/wujud_room.ino.bin" "http://$target/update" >/dev/null
      echo "sent to $target over Wi-Fi; it reboots into it"
    done
  fi
}

node_ips() {
  [[ -n "${BOARD:-}" ]] && return
  call "$board/status" | grep -o '"ip":"[0-9.]*","node":true' | cut -d'"' -f4
}

status() {
  call "$board/status"
  local ip
  for ip in $(node_ips); do call "http://$ip/status"; done
}

case "${1:-}" in
  flash) flash ;;
  status) status ;;
  set)
    shift
    query="$(IFS='&'; echo "$*")"
    call "$board/set?$query"
    ;;
  reboot) call "$board/reboot" ;;
  *)
    sed -n '2,11p' "$0"
    exit 1
    ;;
esac
