/*
 * wujud ROOM firmware: ESP32s, no router. Every phone, laptop or other board that joins the
 * network becomes one sensing link. Every board runs this same firmware and picks its role:
 *
 *   hub   no network called WUJUD_SSID is on the air, so the board opens it and pings every
 *         device that joins (below).
 *   node  the network is there, so the board joins it. The hub then pings it like any device, and
 *         the node itself measures the hub's pings to it and every frame other devices send the
 *         hub (promiscuous), tagged <node mac>><sender mac or "hub">|P,... It sends its lines to the
 *         hub over UDP RELAY_PORT, and the hub passes them on with its own. A node that loses
 *         the hub for NODE_LOST_MS reboots and chooses again.
 *
 * Boards switched on together can both open the network. A hub keeps looking for another one
 * during its first HUB_RIVAL_CHECKS checks; the one with the higher MAC reboots and becomes a node.
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
#include <HTTPClient.h>
#include <Preferences.h>
#include <esp_wifi.h>
#include "ping/ping_sock.h"
#include "lwip/ip_addr.h"
#include "lwip/sockets.h"
#include "ota_key.h"
#if __has_include("home_wifi.h")
#include "home_wifi.h"
#define SHARE_INTERNET 1
#else
#define WUJUD_PASSWORD nullptr
#endif

#define WUJUD_SSID "wujud"
#define DEFAULT_CHANNEL 9
#define DEFAULT_PING_HZ 100
#define LED_PIN 2
#define CSI_PORT 5005
#define RELAY_PORT 5006
#define MAX_DEVICES 8
#define MAX_LISTENERS 4
#define LISTENER_TIMEOUT_MS 5000
#define NODE_LOST_MS 15000
#define RELAY_FLUSH_MS 20
#define HUB_RIVAL_CHECK_MS 30000
#define HUB_RIVAL_CHECKS 6

enum Role { HUB, NODE };

/// Without a ping_len setting, each link keeps the length most of its first short frames had: the
/// pings, 100 a second. A node's relayed lines are long (and all land in the last bucket), so only
/// lengths under LEARN_BELOW count. Mixed in, frames of another length get another gain and look like motion.
#define LEARN_FROM_FRAMES 300
#define LEARN_BELOW 256
struct LearnedLength {
  uint16_t seen;
  int length;
};

struct Device {
  bool connected;
  bool node;
  uint8_t mac[6];
  esp_ip4_addr_t ip;
  esp_ping_handle_t ping;
  volatile uint32_t csi_lines;
  volatile uint32_t rejected;
  uint32_t csi_per_second;
  uint32_t rejected_per_second;
  uint16_t lengths[512];
  LearnedLength ping_len;
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

static Role role;
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
static uint8_t own_mac[6], hub_mac[6], router_mac[6];
static volatile uint32_t node_csi_lines, node_rejected;
static uint32_t node_csi_per_second, node_rejected_per_second;
static uint16_t node_lengths[512];
static LearnedLength node_ping_len;
static uint32_t node_lost_since_ms;

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

static void queue_csi_line(const char *tag, const wifi_csi_info_t *info) {
  CsiLine line;
  int n = snprintf(line.text, sizeof(line.text), "%s|P,%d,%lu,", tag, info->rx_ctrl.rssi,
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

/// One kind of frame per link: 802.11n without STBC, and the ping length when ping_len is set.
/// Legacy frames give a different CSI shape, and STBC from a two-antenna sender changes shape packet
/// by packet; mixed in, either looks like motion.
static bool usable_frame(const wifi_csi_info_t *info, const LearnedLength &learned) {
  if (info->rx_ctrl.sig_mode != 1 || info->rx_ctrl.stbc) return false;
  int length = settings.ping_len ? settings.ping_len : learned.length;
  return !length || info->rx_ctrl.sig_len == length;
}

static int most_common_short(const uint16_t *lengths) {
  int most = 0;
  for (int len = 1; len < LEARN_BELOW; len++)
    if (lengths[len] > lengths[most]) most = len;
  return most;
}

static void learn_length(LearnedLength &learned, uint16_t *lengths, const wifi_csi_info_t *info) {
  if (learned.length || info->rx_ctrl.sig_mode != 1 || info->rx_ctrl.stbc) return;
  if (info->rx_ctrl.sig_len >= LEARN_BELOW || ++learned.seen < LEARN_FROM_FRAMES) return;
  learned.length = most_common_short(lengths);
  report("I,wujud-room,keeping frames of %d bytes", learned.length);
}

/// Learnt too early (a board that joined before its pings flowed kept a length nothing was sent
/// at), a link would stay empty for good: once a second, a length that has clearly taken over wins.
static void relearn_length(LearnedLength &learned, const uint16_t *lengths) {
  if (!learned.length) return;
  int most = most_common_short(lengths);
  if (most != learned.length && lengths[most] > 2 * lengths[learned.length] + LEARN_FROM_FRAMES) {
    learned.length = most;
    report("I,wujud-room,now keeping frames of %d bytes", most);
  }
}

static void hub_csi_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf) return;
  int device = info->len >= 128 ? device_with_mac(info->mac) : -1;
  if (device < 0) {
    others++;
    others_ht += info->rx_ctrl.sig_mode == 1;
    memcpy(last_other, info->mac, 6);
    return;
  }
  Device &d = devices[device];
  if (info->rx_ctrl.sig_mode == 1) {
    uint16_t &seen = d.lengths[min((int)info->rx_ctrl.sig_len, 511)];
    if (seen < 65535) seen++;
  }
  learn_length(d.ping_len, d.lengths, info);
  if (!usable_frame(info, d.ping_len)) {
    devices[device].rejected++;
    return;
  }
  devices[device].csi_lines++;
  char tag[18];
  snprintf(tag, sizeof(tag), MACSTR, MAC2STR(info->mac));
  queue_csi_line(tag, info);
}

/// The hub's frames to this node, and every other device's frames to the hub.
static void node_csi_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || info->len < 128) return;
  bool from_hub = memcmp(info->mac, hub_mac, 6) == 0;
  bool hub_to_me = from_hub && memcmp(info->dmac, own_mac, 6) == 0;
  bool other_to_hub = !from_hub && memcmp(info->dmac, hub_mac, 6) == 0 && memcmp(info->mac, own_mac, 6) != 0;
  if (!hub_to_me && !other_to_hub) return;
  if (hub_to_me && info->rx_ctrl.sig_mode == 1 && node_lengths[min((int)info->rx_ctrl.sig_len, 511)] < 65535) {
    node_lengths[min((int)info->rx_ctrl.sig_len, 511)]++;
    learn_length(node_ping_len, node_lengths, info);
  }
  if (!usable_frame(info, node_ping_len)) {
    node_rejected++;
    return;
  }
  node_csi_lines++;
  char tag[40];
  if (hub_to_me)
    snprintf(tag, sizeof(tag), MACSTR ">hub", MAC2STR(own_mac));
  else
    snprintf(tag, sizeof(tag), MACSTR ">" MACSTR, MAC2STR(own_mac), MAC2STR(info->mac));
  queue_csi_line(tag, info);
}

static int open_socket(int port) {
  int s = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
  sockaddr_in local = {};
  local.sin_family = AF_INET;
  local.sin_port = htons(port);
  bind(s, (const sockaddr *)&local, sizeof(local));
  fcntl(s, F_SETFL, O_NONBLOCK);
  return s;
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

static void send_to_listeners(const char *text, int len) {
  uint32_t now = millis();
  for (const Listener &l : listeners)
    if (l.last_heard_ms && now - l.last_heard_ms < LISTENER_TIMEOUT_MS)
      sendto(udp_socket, text, len, 0, (const sockaddr *)&l.address, sizeof(l.address));
}

/// USB serial carries at most ~90 KB/s; three boards make about twice that. A line that does not fit
/// is left out of the serial copy instead of stalling the network copy behind it.
static void write_serial(const char *text, int len) {
  if (Serial.availableForWrite() >= len) Serial.write((const uint8_t *)text, len);
}

/// Once boards are measuring each other, the router is no longer pinged: it answers in 802.11b
/// (no CSI), and 100 pings a second on its channel take airtime the boards' links need.
static void mark_node(in_addr_t ip) {
  for (Device &d : devices)
    if (d.connected && d.ip.addr == ip) d.node = true;
  int router = device_with_mac(router_mac);
  if (router >= 0 && devices[router].ping) {
    esp_ping_stop(devices[router].ping);
    esp_ping_delete_session(devices[router].ping);
    devices[router].ping = nullptr;
  }
}

/// Node lines arrive as datagrams of whole lines; they go out unchanged, like the hub's own.
static void pass_on_relayed_lines(int relay_socket) {
  static char batch[1500];
  sockaddr_in sender;
  socklen_t sender_len = sizeof(sender);
  int len;
  while ((len = recvfrom(relay_socket, batch, sizeof(batch), 0, (sockaddr *)&sender, &sender_len)) > 0) {
    mark_node(sender.sin_addr.s_addr);
    write_serial(batch, len);
    send_to_listeners(batch, len);
    sender_len = sizeof(sender);
  }
}

/// The hub's own lines go out batched like a node's: one datagram per line, 200 a second, is more
/// than the network stack keeps up with, and most were dropped.
static void hub_send_lines(void *) {
  udp_socket = open_socket(CSI_PORT);
  int relay_socket = open_socket(RELAY_PORT);
  static char batch[1400];
  int used = 0;
  uint32_t first_line_ms = 0;
  CsiLine line;
  for (;;) {
    take_listener_requests();
    pass_on_relayed_lines(relay_socket);
    // Every waiting line, not one per round: with two nodes relaying, one a round let the queue
    // overflow and the hub's own links lost most of their lines.
    TickType_t wait = pdMS_TO_TICKS(2);
    while (xQueueReceive(outgoing, &line, wait) == pdTRUE) {
      wait = 0;
      write_serial(line.text, line.len);
      if (used + line.len > (int)sizeof(batch)) {
        send_to_listeners(batch, used);
        used = 0;
      }
      if (!used) first_line_ms = millis();
      memcpy(batch + used, line.text, line.len);
      used += line.len;
    }
    if (used && millis() - first_line_ms >= RELAY_FLUSH_MS) {
      send_to_listeners(batch, used);
      used = 0;
    }
  }
}

/// Lines are batched into datagrams of up to one Wi-Fi frame, flushed at least every RELAY_FLUSH_MS.
static void node_send_lines(void *) {
  udp_socket = open_socket(RELAY_PORT);
  sockaddr_in hub = {};
  hub.sin_family = AF_INET;
  hub.sin_port = htons(RELAY_PORT);
  hub.sin_addr.s_addr = inet_addr("192.168.4.1");
  static char batch[1400];
  int used = 0;
  uint32_t first_line_ms = 0;
  CsiLine line;
  for (;;) {
    bool got = xQueueReceive(outgoing, &line, pdMS_TO_TICKS(5)) == pdTRUE;
    if (got) write_serial(line.text, line.len);
    bool full = got && used + line.len > (int)sizeof(batch);
    bool stale = used && millis() - first_line_ms >= RELAY_FLUSH_MS;
    if (full || stale) {
      sendto(udp_socket, batch, used, 0, (const sockaddr *)&hub, sizeof(hub));
      used = 0;
    }
    if (!got) continue;
    if (!used) first_line_ms = millis();
    memcpy(batch + used, line.text, line.len);
    used += line.len;
  }
}

static esp_ping_handle_t start_ping(esp_ip4_addr_t ip) {
  esp_ping_config_t cfg = ESP_PING_DEFAULT_CONFIG();
  cfg.target_addr.type = IPADDR_TYPE_V4;
  cfg.target_addr.u_addr.ip4.addr = ip.addr;
  cfg.count = ESP_PING_COUNT_INFINITE;
  cfg.interval_ms = 1000 / settings.ping_hz;
  // The next ping waits for this ping's reply: a lost reply must not cost a whole second.
  cfg.timeout_ms = cfg.interval_ms;
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
    d.ip = ip;
    d.node = false;
    memset(d.lengths, 0, sizeof(d.lengths));
    d.ping_len = {};
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


static void on_hub_wifi_event(arduino_event_t *event) {
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

static void on_node_wifi_event(arduino_event_t *event) {
  if (event->event_id == ARDUINO_EVENT_WIFI_STA_GOT_IP) {
    node_lost_since_ms = 0;
    digitalWrite(LED_PIN, HIGH);
  } else if (event->event_id == ARDUINO_EVENT_WIFI_STA_DISCONNECTED) {
    if (!node_lost_since_ms) node_lost_since_ms = millis();
    digitalWrite(LED_PIN, LOW);
  }
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

static void serve_node_status() {
  char json[260];
  snprintf(json, sizeof(json),
           "{\"role\":\"node\",\"uptime_s\":%lu,\"mac\":\"" MACSTR "\",\"hub\":\"" MACSTR
           "\",\"rssi\":%d,\"ping_len\":%d,\"csi_per_s\":%lu,\"rejected_per_s\":%lu,\"free_heap\":%lu}\n",
           (unsigned long)(millis() / 1000), MAC2STR(own_mac), MAC2STR(hub_mac), WiFi.RSSI(), settings.ping_len,
           (unsigned long)node_csi_per_second, (unsigned long)node_rejected_per_second,
           (unsigned long)ESP.getFreeHeap());
  web.send(200, "application/json", json);
}

static void serve_status() {
  if (role == NODE) return serve_node_status();
  String json = "{\"role\":\"hub\",\"uptime_s\":" + String(millis() / 1000) + ",\"channel\":" + String(settings.channel) +
                ",\"ping_hz\":" + String(settings.ping_hz) +
                ",\"no_11b\":" + String(settings.no_11b) + ",\"ping_len\":" + String(settings.ping_len) + ",\"free_heap\":" + String(ESP.getFreeHeap()) +
                ",\"devices\":[";
  bool first = true;
  for (const Device &d : devices) {
    if (!d.connected) continue;
    char entry[180];
    snprintf(entry, sizeof(entry),
             "%s{\"mac\":\"" MACSTR "\",\"ip\":\"" IPSTR "\",\"node\":%s,\"csi_per_s\":%lu,\"rejected_per_s\":%lu,\"lengths\":{",
             first ? "" : ",", MAC2STR(d.mac), IP2STR(&d.ip), d.node ? "true" : "false",
             (unsigned long)d.csi_per_second, (unsigned long)d.rejected_per_second);
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

/// Tells the access point this board speaks only 802.11g/n, so it has no 802.11b rate to answer with.
static void refuse_11b() {
  esp_wifi_stop();
  esp_wifi_set_protocol(WIFI_IF_STA, WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N);
  esp_err_t err = esp_wifi_config_11b_rate(WIFI_IF_STA, true);
  esp_wifi_start();
  report("I,wujud-room,no_11b=1 (%s)", esp_err_to_name(err));
}

#ifdef SHARE_INTERNET
/// Joins the home router; returns its channel, or 0 when it cannot be reached.
static int join_home() {
  if (settings.no_11b) refuse_11b();
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

/// Strongest network called WUJUD_SSID on the air; false when there is none.
static bool find_hub(uint8_t *bssid, int *channel) {
  int found = WiFi.scanNetworks(false, false, false, 120, 0, WUJUD_SSID);
  int best = -1;
  for (int i = 0; i < found; i++)
    if (WiFi.SSID(i) == WUJUD_SSID && (best < 0 || WiFi.RSSI(i) > WiFi.RSSI(best))) best = i;
  if (best >= 0) {
    memcpy(bssid, WiFi.BSSID(best), 6);
    *channel = WiFi.channel(best);
  }
  WiFi.scanDelete();
  return best >= 0;
}

/// Another hub on this hub's channel with a lower MAC: that one stays, this one becomes a node.
static bool rival_hub_wins() {
  uint8_t ap_mac[6];
  WiFi.softAPmacAddress(ap_mac);
  int found = WiFi.scanNetworks(false, false, false, 120, settings.channel, WUJUD_SSID);
  bool rival = false;
  for (int i = 0; i < found; i++)
    rival |= WiFi.SSID(i) == WUJUD_SSID && memcmp(WiFi.BSSID(i), ap_mac, 6) < 0;
  WiFi.scanDelete();
  return rival;
}

/// The hub's ping_len, so a node keeps the same frames the hub keeps.
static void copy_hub_ping_len() {
  HTTPClient http;
  http.setTimeout(3000);
  if (!http.begin("http://192.168.4.1/status") || http.GET() != 200) return;
  String json = http.getString();
  http.end();
  int at = json.indexOf("\"ping_len\":");
  if (at >= 0) settings.ping_len = json.substring(at + 11).toInt();
}

static void start_csi(wifi_csi_cb_t callback) {
  wifi_csi_config_t cfg = {};
  cfg.lltf_en = true;
  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(callback, NULL);
  esp_wifi_set_csi(true);
}

static void start_hub() {
  role = HUB;
  WiFi.onEvent(on_hub_wifi_event);
  WiFi.mode(WIFI_AP_STA);
  xTaskCreate(hub_send_lines, "send_lines", 4096, nullptr, 5, nullptr);
#ifdef SHARE_INTERNET
  int home_channel = join_home();
  if (home_channel) {
    settings.channel = home_channel;
    open_network(home_channel);
  } else {
    report("I,wujud-room,cannot reach \"%s\": no internet for the devices", HOME_SSID);
    // Retrying in the background would keep the radio scanning other channels and drop the pings.
    WiFi.setAutoReconnect(false);
    WiFi.disconnect();
    WiFi.softAP(WUJUD_SSID, WUJUD_PASSWORD, settings.channel, 0, MAX_DEVICES);
  }
#else
  WiFi.softAP(WUJUD_SSID, WUJUD_PASSWORD, settings.channel, 0, MAX_DEVICES);
#endif
  // OFDM only from the hub too, so the nodes can measure every ping it sends them.
  esp_wifi_config_11b_rate(WIFI_IF_AP, true);
  esp_wifi_set_ps(WIFI_PS_NONE);
  start_csi(hub_csi_cb);
  report("I,wujud-room,hub,waiting for devices on Wi-Fi \"%s\",ch=%d,ping_hz=%d", WUJUD_SSID, settings.channel,
         settings.ping_hz);
}

static void start_node(const uint8_t *bssid, int channel) {
  role = NODE;
  memcpy(hub_mac, bssid, 6);
  WiFi.macAddress(own_mac);
  WiFi.onEvent(on_node_wifi_event);
  xTaskCreate(node_send_lines, "send_lines", 4096, nullptr, 5, nullptr);
  refuse_11b();
  WiFi.begin(WUJUD_SSID, WUJUD_PASSWORD, channel, bssid);
  for (uint32_t start = millis(); WiFi.status() != WL_CONNECTED; delay(100))
    if (millis() - start > NODE_LOST_MS) ESP.restart();
  esp_wifi_set_ps(WIFI_PS_NONE);
  copy_hub_ping_len();
  wifi_promiscuous_filter_t data_only = {WIFI_PROMIS_FILTER_MASK_DATA};
  esp_wifi_set_promiscuous_filter(&data_only);
  esp_wifi_set_promiscuous(true);
  start_csi(node_csi_cb);
  report("I,wujud-room,node " MACSTR ",hub " MACSTR ",ch=%d,ping_len=%d", MAC2STR(own_mac), MAC2STR(hub_mac),
         channel, settings.ping_len);
}

void setup() {
  pinMode(LED_PIN, OUTPUT);
  load_settings();
  Serial.setTxBufferSize(16384);
  Serial.begin(921600);
  outgoing = xQueueCreate(96, sizeof(CsiLine));
  // Boards switched on together wait different times, so usually the first one's network is up
  // before the others look for it.
  delay(esp_random() % 2500);
  WiFi.mode(WIFI_STA);
  uint8_t bssid[6];
  int channel;
  if (find_hub(bssid, &channel))
    start_node(bssid, channel);
  else
    start_hub();

  firmware_update.setup(&web, "wujud", OTA_KEY);
  web.on("/status", serve_status);
  web.on("/set", serve_set);
  web.on("/reboot", serve_reboot);
  web.begin();
}

static void count_last_second() {
  for (Device &d : devices) {
    if (d.connected) relearn_length(d.ping_len, d.lengths);
    d.csi_per_second = d.csi_lines;
    d.rejected_per_second = d.rejected;
    d.csi_lines = 0;
    d.rejected = 0;
  }
  others_per_second = others;
  others_ht_per_second = others_ht;
  others = others_ht = 0;
  relearn_length(node_ping_len, node_lengths);
  node_csi_per_second = node_csi_lines;
  node_rejected_per_second = node_rejected;
  node_csi_lines = node_rejected = 0;
}

void loop() {
  static uint32_t last_count_ms = 0, last_rival_check_ms = 0;
  static int rival_checks = 0;
  web.handleClient();
  uint32_t now = millis();
  if (now - last_count_ms >= 1000) {
    last_count_ms = now;
    count_last_second();
  }
  if (role == NODE && node_lost_since_ms && now - node_lost_since_ms > NODE_LOST_MS) ESP.restart();
  if (role == HUB && rival_checks < HUB_RIVAL_CHECKS && now - last_rival_check_ms >= HUB_RIVAL_CHECK_MS) {
    last_rival_check_ms = now;
    rival_checks++;
    if (rival_hub_wins()) {
      report("I,wujud-room,another hub has a lower MAC: rebooting as a node");
      delay(200);
      ESP.restart();
    }
  }
  delay(2);
}
