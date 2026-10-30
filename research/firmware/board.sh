#!/usr/bin/env bash
# The room board, from the laptop. Over USB when it is plugged in, otherwise over Wi-Fi
# (the laptop joined to the board's network "wujud").
#   board.sh flash                     build research/firmware/wujud_room and send it
#                                      (OVER_WIFI=1 sends it over Wi-Fi even when plugged in)
#   board.sh status                    what the board is doing
#   board.sh set ping_hz=200 channel=6 change settings (the board reboots)
#   board.sh reboot
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
sketch="$here/wujud_room"
key_file="$sketch/ota_key.h"
board="http://192.168.4.1"

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
    # The update page refuses uploads whose Origin is not the board itself (CSRF guard).
    call -H "Origin: $board" -F "firmware=@$build/wujud_room.ino.bin" "$board/update" >/dev/null
    echo "sent over Wi-Fi; the board reboots into it"
  fi
}

case "${1:-}" in
  flash) flash ;;
  status) call "$board/status" ;;
  set)
    shift
    query="$(IFS='&'; echo "$*")"
    call "$board/set?$query"
    ;;
  reboot) call "$board/reboot" ;;
  *)
    sed -n '2,9p' "$0"
    exit 1
    ;;
esac
