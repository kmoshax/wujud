/*
 * wujud ROOM firmware: one ESP32, no router. Every phone or laptop that joins its Wi-Fi
 * becomes one sensing link.
 *
 * The board opens its own Wi-Fi network (WUJUD_SSID) and pings every device that joins,
 * ping_hz times per second each (100 unless set). Every reply is OFDM, so it carries CSI. One line per packet,
 * tagged with the device that sent it:
 *     <device mac>|P,<rssi>,<rx timestamp us>,<128 hex chars = 64 subcarriers x [imag, real] int8>
 * Lines go over USB, and as UDP datagrams to whoever sent anything to CSI_PORT in the last
 * LISTENER_TIMEOUT_MS (asking first lets the reply through the laptop's firewall).
 * The blue LED is on while at least one device is connected.
 *
 * With home_wifi.h (HOME_SSID, HOME_PASSWORD, WUJUD_PASSWORD; not in git), the board also joins the
 * home router and passes the internet on to the devices that join it, so a phone keeps working
 * normally while it is a sensing link. Its own network then needs WUJUD_PASSWORD and sits on the
 * router's channel (one radio, one channel). The router itself is then pinged too, as one more
 * link, so its CSI can be judged against the phones'. With no_11b=1 (the default) the board tells
 * the router it speaks only 802.11g/n, so that the router has no 802.11b rate to answer with.
 *
 * Remote control over Wi-Fi (research/firmware/board.sh wraps these):
 *     GET  /status                       what the board is doing, as JSON
 *     GET  /set?ping_hz=200&channel=6    change settings (kept across reboots), then reboot
 *     GET  /set?no_11b=0                 let the router answer in 802.11b again (to compare)
 *     GET  /set?ping_len=83              keep only frames of this length (the ping replies), so a video
 *                                        playing on the phone keeps it awake without mixing its own
 *                                        frames into the CSI; 0 keeps every length. /status lists the
 *                                        commonest lengths per device.
 *     GET  /reboot
 *     POST /update                       new firmware
 * All but /status need user "wujud" and the key in ota_key.h. Status lines (I,...) also go to
 * the UDP listeners, so a board on a charger is as visible as one on USB.
 *
 * Why not the home router: cheap routers answer the ESP32 in 802.11b (DSSS), which has no
 * subcarriers and so no CSI, and the classic ESP32 cannot refuse 802.11b.
 *
 * Arduino-ESP32 core 3.x. Serial 921600.
 */
#include <WiFi.h>
#include <WebServer.h>
#include <HTTPUpdateServer.h>
#include <Preferences.h>
#include <esp_wifi.h>
#include "ping/ping_sock.h"
#include "lwip/ip_addr.h"
#include "lwip/sockets.h"
#include "ota_key.h"
#if __has_include("home_wifi.h")
#include "home_wifi.h"
#define SHARE_INTERNET 1
#endif

#define WUJUD_SSID "wujud"
#define DEFAULT_CHANNEL 9
#define DEFAULT_PING_HZ 100
#define LED_PIN 2
#define CSI_PORT 5005
#define MAX_DEVICES 4
#define MAX_LISTENERS 4
#define LISTENER_TIMEOUT_MS 5000

struct Device {
  bool connected;
  uint8_t mac[6];
  esp_ping_handle_t ping;
  volatile uint32_t csi_lines;
  volatile uint32_t rejected;
  uint32_t csi_per_second;
  uint32_t rejected_per_second;
  uint16_t lengths[512];
};

struct Settings {
  int channel;
  int ping_hz;
  int no_11b;
  int ping_len;
};

struct Listener {
  sockaddr_in address;
  uint32_t last_heard_ms;
};

struct CsiLine {
  int len;
  char text[320];
};

static Device devices[MAX_DEVICES];
static Listener listeners[MAX_LISTENERS];
// The CSI callback runs in the Wi-Fi task; sending from there would stall it, so lines go through a queue.
static QueueHandle_t outgoing;
static int udp_socket;
static WebServer web(80);
static HTTPUpdateServer firmware_update;
static Preferences stored;
static Settings settings;
static const char HEX_[] = "0123456789abcdef";
// CSI from senders that are not links (neighbours, other routers), counted to see what else is on the air.
static volatile uint32_t others, others_ht;
static uint8_t last_other[6];
static uint32_t others_per_second, others_ht_per_second;

static int device_with_mac(const uint8_t *mac) {
  for (int i = 0; i < MAX_DEVICES; i++)
    if (devices[i].connected && memcmp(devices[i].mac, mac, 6) == 0) return i;
  return -1;
}

static int connected_devices() {
  int count = 0;
  for (const Device &d : devices) count += d.connected;
  return count;
}

/// Status lines go where the CSI goes: USB and every UDP listener.
static void report(const char *format, ...) {
  CsiLine line;
  va_list args;
  va_start(args, format);
  line.len = vsnprintf(line.text, sizeof(line.text) - 1, format, args);
  va_end(args);
  line.len = min(line.len, (int)sizeof(line.text) - 2);
  line.text[line.len++] = '\n';
  xQueueSend(outgoing, &line, 0);
}

static void csi_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;
  int device = info->len >= 128 ? device_with_mac(info->mac) : -1;
  if (device < 0) {
    others++;
    others_ht += info->rx_ctrl.sig_mode == 1;
    memcpy(last_other, info->mac, 6);
    return;
  }
  // Only one kind of frame per link: 802.11n without STBC. Legacy frames give a different CSI shape,
  // and STBC from a two-antenna sender changes shape packet by packet; mixed in, either looks like motion.
  if (info->rx_ctrl.sig_mode != 1 || info->rx_ctrl.stbc) {
    devices[device].rejected++;
    return;
  }
  uint16_t &seen = devices[device].lengths[min((int)info->rx_ctrl.sig_len, 511)];
  if (seen < 65535) seen++;
  if (settings.ping_len && info->rx_ctrl.sig_len != settings.ping_len) {
    devices[device].rejected++;
    return;
  }
  devices[device].csi_lines++;
  CsiLine line;
  int n = snprintf(line.text, sizeof(line.text), MACSTR "|P,%d,%lu,", MAC2STR(info->mac), info->rx_ctrl.rssi,
                   (unsigned long)info->rx_ctrl.timestamp);
  const uint8_t *b = (const uint8_t *)info->buf;
  for (int i = 0; i < 128 && n < (int)sizeof(line.text) - 3; i++) {
    line.text[n++] = HEX_[b[i] >> 4];
    line.text[n++] = HEX_[b[i] & 15];
  }
  line.text[n++] = '\n';
  line.len = n;
  xQueueSend(outgoing, &line, 0);
}

static void open_socket() {
  udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  sockaddr_in local = {};
  local.sin_family = AF_INET;
  local.sin_port = htons(CSI_PORT);
  bind(udp_socket, (const sockaddr *)&local, sizeof(local));
  fcntl(udp_socket, F_SETFL, O_NONBLOCK);
}

static void take_listener_requests() {
  char request[32];
  sockaddr_in sender;
  socklen_t sender_len = sizeof(sender);
  while (recvfrom(udp_socket, request, sizeof(request), 0, (sockaddr *)&sender, &sender_len) >= 0) {
    Listener *slot = &listeners[0];
    for (Listener &l : listeners) {
      bool same = l.address.sin_addr.s_addr == sender.sin_addr.s_addr && l.address.sin_port == sender.sin_port;
      if (same) {
        slot = &l;
        break;
      }
      if (l.last_heard_ms < slot->last_heard_ms) slot = &l;
    }
    slot->address = sender;
    slot->last_heard_ms = millis();
    sender_len = sizeof(sender);
  }
}

static void send_lines(void *) {
  open_socket();
  CsiLine line;
  for (;;) {
    take_listener_requests();
    if (xQueueReceive(outgoing, &line, pdMS_TO_TICKS(100)) != pdTRUE) continue;
    Serial.write((const uint8_t *)line.text, line.len);
    uint32_t now = millis();
    for (const Listener &l : listeners)
      if (l.last_heard_ms && now - l.last_heard_ms < LISTENER_TIMEOUT_MS)
        sendto(udp_socket, line.text, line.len, 0, (const sockaddr *)&l.address, sizeof(l.address));
  }
}

static esp_ping_handle_t start_ping(esp_ip4_addr_t ip) {
  esp_ping_config_t cfg = ESP_PING_DEFAULT_CONFIG();
  cfg.target_addr.type = IPADDR_TYPE_V4;
  cfg.target_addr.u_addr.ip4.addr = ip.addr;
  cfg.count = ESP_PING_COUNT_INFINITE;
  cfg.interval_ms = 1000 / settings.ping_hz;
  cfg.timeout_ms = 1000;
  cfg.data_size = 1;
  esp_ping_callbacks_t cbs = {};
  esp_ping_handle_t ping = nullptr;
  if (esp_ping_new_session(&cfg, &cbs, &ping) == ESP_OK) esp_ping_start(ping);
  return ping;
}

static void device_joined(const uint8_t *mac, esp_ip4_addr_t ip) {
  if (device_with_mac(mac) >= 0) return;
  for (Device &d : devices) {
    if (d.connected) continue;
    memcpy(d.mac, mac, 6);
    d.ping = start_ping(ip);
    d.connected = true;
    digitalWrite(LED_PIN, HIGH);
    report("I,wujud-room,joined=" MACSTR ",devices=%d", MAC2STR(mac), connected_devices());
    return;
  }
}

static void device_left(const uint8_t *mac) {
  int index = device_with_mac(mac);
  if (index < 0) return;
  Device &d = devices[index];
  d.connected = false;
  if (d.ping) {
    esp_ping_stop(d.ping);
    esp_ping_delete_session(d.ping);
    d.ping = nullptr;
  }
  digitalWrite(LED_PIN, connected_devices() > 0);
  report("I,wujud-room,left=" MACSTR ",devices=%d", MAC2STR(mac), connected_devices());
}

static uint8_t router_mac[6];

static void on_wifi_event(arduino_event_t *event) {
  if (event->event_id == ARDUINO_EVENT_WIFI_AP_STAIPASSIGNED)
    device_joined(event->event_info.wifi_ap_staipassigned.mac, event->event_info.wifi_ap_staipassigned.ip);
  else if (event->event_id == ARDUINO_EVENT_WIFI_AP_STADISCONNECTED)
    device_left(event->event_info.wifi_ap_stadisconnected.mac);
  else if (event->event_id == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    memcpy(router_mac, WiFi.BSSID(), 6);
    device_joined(router_mac, event->event_info.got_ip.ip_info.gw);
  } else if (event->event_id == ARDUINO_EVENT_WIFI_STA_DISCONNECTED)
    device_left(router_mac);
}

static void load_settings() {
  stored.begin("wujud", false);
  settings.channel = stored.getInt("channel", DEFAULT_CHANNEL);
  settings.ping_hz = stored.getInt("ping_hz", DEFAULT_PING_HZ);
  settings.no_11b = stored.getInt("no_11b", 1);
  settings.ping_len = stored.getInt("ping_len", 0);
}

static bool authorised() {
  if (web.authenticate("wujud", OTA_KEY)) return true;
  web.requestAuthentication();
  return false;
}

static void reboot_soon() {
  web.client().flush();
  delay(300);
  ESP.restart();
}

static void serve_status() {
  String json = "{\"uptime_s\":" + String(millis() / 1000) + ",\"channel\":" + String(settings.channel) +
                ",\"ping_hz\":" + String(settings.ping_hz) +
                ",\"no_11b\":" + String(settings.no_11b) + ",\"ping_len\":" + String(settings.ping_len) + ",\"free_heap\":" + String(ESP.getFreeHeap()) +
                ",\"devices\":[";
  bool first = true;
  for (const Device &d : devices) {
    if (!d.connected) continue;
    char entry[120];
    snprintf(entry, sizeof(entry), "%s{\"mac\":\"" MACSTR "\",\"csi_per_s\":%lu,\"rejected_per_s\":%lu,\"lengths\":{",
             first ? "" : ",", MAC2STR(d.mac), (unsigned long)d.csi_per_second, (unsigned long)d.rejected_per_second);
    json += entry;
    int chosen[5] = {-1, -1, -1, -1, -1};
    for (int shown = 0; shown < 5; shown++) {
      for (int len = 0; len < 512; len++) {
        bool taken = false;
        for (int c : chosen) taken |= c == len;
        if (!taken && d.lengths[len] && (chosen[shown] < 0 || d.lengths[len] > d.lengths[chosen[shown]]))
          chosen[shown] = len;
      }
      if (chosen[shown] < 0) break;
      json += String(shown ? "," : "") + "\"" + String(chosen[shown]) + "\":" + String(d.lengths[chosen[shown]]);
    }
    json += "}}";
    first = false;
  }
  char tail[120];
  snprintf(tail, sizeof(tail), "],\"others_per_s\":%lu,\"others_ht_per_s\":%lu,\"last_other\":\"" MACSTR "\"}\n",
           (unsigned long)others_per_second, (unsigned long)others_ht_per_second, MAC2STR(last_other));
  web.send(200, "application/json", json + tail);
}

static void serve_set() {
  if (!authorised()) return;
  bool changed = false;
  if (web.hasArg("channel")) {
    int channel = web.arg("channel").toInt();
    if (channel < 1 || channel > 13) return web.send(400, "text/plain", "channel must be 1-13\n");
    stored.putInt("channel", channel);
    changed = true;
  }
  if (web.hasArg("ping_len")) {
    stored.putInt("ping_len", web.arg("ping_len").toInt());
    changed = true;
  }
  if (web.hasArg("no_11b")) {
    stored.putInt("no_11b", web.arg("no_11b").toInt() ? 1 : 0);
    changed = true;
  }
  if (web.hasArg("ping_hz")) {
    int hz = web.arg("ping_hz").toInt();
    if (hz < 10 || hz > 500) return web.send(400, "text/plain", "ping_hz must be 10-500\n");
    stored.putInt("ping_hz", hz);
    changed = true;
  }
  if (!changed) return web.send(400, "text/plain", "nothing to set (channel, ping_hz, no_11b, ping_len)\n");
  web.send(200, "text/plain", "saved, rebooting\n");
  reboot_soon();
}

static void serve_reboot() {
  if (!authorised()) return;
  web.send(200, "text/plain", "rebooting\n");
  reboot_soon();
}

#ifdef SHARE_INTERNET
/// Joins the home router; returns its channel, or 0 when it cannot be reached.
static int join_home() {
  if (settings.no_11b) {
    esp_wifi_stop();
    esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
    esp_err_t err = esp_wifi_config_11b_rate(WIFI_IF_STA, true);
    esp_wifi_start();
    report("I,wujud-room,no_11b=1 (%s)", esp_err_to_name(err));
  }
  WiFi.begin(HOME_SSID, HOME_PASSWORD);
  for (uint32_t start = millis(); WiFi.status() != WL_CONNECTED && millis() - start < 20000;) delay(100);
  return WiFi.status() == WL_CONNECTED ? WiFi.channel() : 0;
}

static void open_network(int channel) {
  WiFi.softAPConfig(IPAddress(192, 168, 4, 1), IPAddress(192, 168, 4, 1), IPAddress(255, 255, 255, 0),
                    IPAddress(192, 168, 4, 2), WiFi.dnsIP());
  WiFi.softAP(WUJUD_SSID, WUJUD_PASSWORD, channel, 0, MAX_DEVICES);
  WiFi.AP.enableNAPT(true);
  report("I,wujud-room,internet via \"%s\" at %s, router " MACSTR, HOME_SSID, WiFi.localIP().toString().c_str(),
         MAC2STR(router_mac));
}
#endif

void setup() {
  pinMode(LED_PIN, OUTPUT);
  load_settings();
  Serial.setTxBufferSize(16384);
  Serial.begin(921600);
  outgoing = xQueueCreate(48, sizeof(CsiLine));
  WiFi.onEvent(on_wifi_event);
#ifdef SHARE_INTERNET
  WiFi.mode(WIFI_AP_STA);
  xTaskCreate(send_lines, "send_lines", 4096, nullptr, 5, nullptr);
  int home_channel = join_home();
  if (home_channel) {
    settings.channel = home_channel;
    open_network(home_channel);
  } else {
    report("I,wujud-room,cannot reach \"%s\": no internet for the devices", HOME_SSID);
    WiFi.softAP(WUJUD_SSID, WUJUD_PASSWORD, settings.channel, 0, MAX_DEVICES);
  }
#else
  WiFi.mode(WIFI_AP);
  xTaskCreate(send_lines, "send_lines", 4096, nullptr, 5, nullptr);
  WiFi.softAP(WUJUD_SSID, nullptr, settings.channel, 0, MAX_DEVICES);
#endif
  esp_wifi_set_ps(WIFI_PS_NONE);

  wifi_csi_config_t cfg = {};
  cfg.lltf_en = true;
  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(csi_cb, NULL);
  esp_wifi_set_csi(true);

  firmware_update.setup(&web, "wujud", OTA_KEY);
  web.on("/status", serve_status);
  web.on("/set", serve_set);
  web.on("/reboot", serve_reboot);
  web.begin();
  report("I,wujud-room,waiting for devices on Wi-Fi \"%s\",ch=%d,ping_hz=%d", WUJUD_SSID, settings.channel,
         settings.ping_hz);
}

void loop() {
  static uint32_t last_count_ms = 0;
  web.handleClient();
  if (millis() - last_count_ms >= 1000) {
    last_count_ms = millis();
    for (Device &d : devices) {
      d.csi_per_second = d.csi_lines;
      d.rejected_per_second = d.rejected;
      d.csi_lines = 0;
      d.rejected = 0;
    }
    others_per_second = others;
    others_ht_per_second = others_ht;
    others = others_ht = 0;
  }
  delay(2);
}
