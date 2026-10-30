/*
 * HallSense node firmware — ESP32 / ESP32-S3 (Arduino-ESP32 core 3.x, ESP-IDF 5.x)
 *
 * Every node:
 *   - broadcasts a tiny ESP-NOW frame FS times per second at HT20 MCS0 (OFDM, so it yields CSI)
 *   - captures the L-LTF CSI of every frame it hears from other HallSense nodes
 *   - turns each packet into 13 subcarrier-group levels in absolute dB
 *       a_k = 10 log10(p_k / mean p) + RSSI       (same math as hallsense/features.py)
 *   - once per second broadcasts a REPORT with mean/std per source node
 * The SINK (any node with its USB plugged into the laptop) prints every
 * report it hears, plus its own, as CSV lines on Serial.
 *
 * Set NODE_ID below (0..31) before flashing each board. All boards: same CHANNEL.
 *
 * Status: written against the documented ESP-IDF CSI / ESP-NOW APIs but NOT yet
 * tested on hardware. First thing to check on the bench: "R," lines appear
 * on the sink and n is ~FS per source.
 */
#include <WiFi.h>
#include <esp_wifi.h>
#include <esp_now.h>
#include <math.h>

#ifndef NODE_ID
#define NODE_ID 0          // <-- change per board
#endif
#define CHANNEL 6
#define FS 20              // beacons per second
#define MAX_NODES 32
#define N_GROUPS 13
#define OUTLIER_DB 8.0f
#define MAGIC 0x48         // 'H'

enum { T_BEACON = 1, T_REPORT = 2 };

typedef struct __attribute__((packed)) {
  uint8_t magic, type, id;
  uint16_t seq;
} beacon_t;

typedef struct __attribute__((packed)) {
  uint8_t src, n;
  int16_t mean_q4[N_GROUPS];   // dB * 4
  uint8_t std_q20[N_GROUPS];   // dB * 20
} rec_t;

typedef struct __attribute__((packed)) {
  uint8_t magic, type, rx, nrec;
  uint16_t seq;
  rec_t rec[5];
} report_t;

static const uint8_t BCAST[6] = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff};

// MAC -> node id, learned from beacons
static uint8_t mac_tab[MAX_NODES][6];
static bool mac_known[MAX_NODES];

// per-source accumulators for the current 1 s window
typedef struct {
  float sum[N_GROUPS], sq[N_GROUPS];
  uint16_t n;
  float ema;
  bool ema_init;
} acc_t;
static acc_t acc[MAX_NODES];
static portMUX_TYPE mux = portMUX_INITIALIZER_UNLOCKED;
static uint16_t seq = 0;

static int id_of_mac(const uint8_t *mac) {
  for (int i = 0; i < MAX_NODES; i++)
    if (mac_known[i] && memcmp(mac_tab[i], mac, 6) == 0) return i;
  return -1;
}

// ---- CSI callback: runs in the Wi-Fi task, keep it short -------------------
static void csi_cb(void *ctx, wifi_csi_info_t *info) {
  if (!info || !info->buf || info->len < 128) return;
  int src = id_of_mac(info->mac);
  if (src < 0 || src == NODE_ID) return;
  const int8_t *b = info->buf;          // L-LTF: 64 x [imag, real]; index 0..31 = k 0..31, 32..63 = k -32..-1
  float p[52];
  float mean = 0;
  int n = 0;
  for (int k = -26; k <= 26; k++) {
    if (k == 0) continue;
    int idx = (k > 0) ? k : 64 + k;
    float im = b[2 * idx], re = b[2 * idx + 1];
    float v = re * re + im * im;
    if (v < 0.5f) v = 0.5f;
    p[n++] = v;
    mean += v;
  }
  // on some chips the first 4 bytes (subcarriers 0 and +1) are invalid: reuse +2 for +1
  if (info->first_word_invalid) { mean -= p[26]; p[26] = p[27]; mean += p[26]; }
  mean /= 52.0f;
  float rssi = info->rx_ctrl.rssi;
  float g[N_GROUPS];
  // order must match SC_IDX = [-26..-1, 1..26]: p[] was filled -26..-1 then 1..26
  for (int m = 0; m < N_GROUPS; m++) {
    float s = 0;
    for (int q = 0; q < 4; q++) s += 10.0f * log10f(p[4 * m + q] / mean);
    g[m] = s / 4.0f + rssi;
  }
  portENTER_CRITICAL(&mux);
  acc_t *a = &acc[src];
  if (!a->ema_init) { a->ema = rssi; a->ema_init = true; }
  if (fabsf(rssi - a->ema) <= OUTLIER_DB) {
    a->ema += 0.1f * (rssi - a->ema);
    for (int m = 0; m < N_GROUPS; m++) { a->sum[m] += g[m]; a->sq[m] += g[m] * g[m]; }
    a->n++;
  }
  portEXIT_CRITICAL(&mux);
}

static void print_rec(int rx, const rec_t *r) {
  Serial.printf("R,%d,%d,%d", rx, r->src, r->n);
  for (int m = 0; m < N_GROUPS; m++) Serial.printf(",%.2f", r->mean_q4[m] / 4.0f);
  for (int m = 0; m < N_GROUPS; m++) Serial.printf(",%.2f", r->std_q20[m] / 20.0f);
  Serial.print("\n");
}

// ---- ESP-NOW receive: learn MACs, and (on the sink) print others' reports ----
static void on_recv(const esp_now_recv_info_t *ri, const uint8_t *data, int len) {
  if (len < 3 || data[0] != MAGIC) return;
  const uint8_t *mac = ri->src_addr;
  if (data[1] == T_BEACON && len >= (int)sizeof(beacon_t)) {
    uint8_t id = data[2];
    if (id < MAX_NODES && !mac_known[id]) {
      memcpy(mac_tab[id], mac, 6);
      mac_known[id] = true;
      Serial.printf("N,%d,%02x:%02x:%02x:%02x:%02x:%02x\n", id, mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    }
  } else if (data[1] == T_REPORT) {
    const report_t *r = (const report_t *)data;
    for (int i = 0; i < r->nrec && i < 5; i++) print_rec(r->rx, &r->rec[i]);
  }
}

static void flush_reports() {
  static acc_t snap[MAX_NODES];
  portENTER_CRITICAL(&mux);
  memcpy(snap, acc, sizeof(acc));
  for (int i = 0; i < MAX_NODES; i++) {
    float e = acc[i].ema; bool ei = acc[i].ema_init;
    memset(&acc[i], 0, sizeof(acc_t));
    acc[i].ema = e; acc[i].ema_init = ei;
  }
  portEXIT_CRITICAL(&mux);

  report_t rep;
  rep.magic = MAGIC; rep.type = T_REPORT; rep.rx = NODE_ID; rep.seq = seq; rep.nrec = 0;
  for (int s = 0; s < MAX_NODES; s++) {
    if (snap[s].n < 3) continue;
    rec_t *r = &rep.rec[rep.nrec];
    r->src = s;
    r->n = snap[s].n > 255 ? 255 : snap[s].n;
    for (int m = 0; m < N_GROUPS; m++) {
      float mu = snap[s].sum[m] / snap[s].n;
      float var = snap[s].sq[m] / snap[s].n - mu * mu;
      float sd = var > 0 ? sqrtf(var) : 0;
      r->mean_q4[m] = (int16_t)lroundf(mu * 4.0f);
      r->std_q20[m] = (uint8_t)(sd * 20.0f > 255 ? 255 : lroundf(sd * 20.0f));
    }
    print_rec(NODE_ID, r);              // local copy (useful when this node is the sink)
    if (++rep.nrec == 5) {
      esp_now_send(BCAST, (uint8_t *)&rep, 6 + rep.nrec * sizeof(rec_t));
      rep.nrec = 0;
    }
  }
  if (rep.nrec) esp_now_send(BCAST, (uint8_t *)&rep, 6 + rep.nrec * sizeof(rec_t));
}

void setup() {
  Serial.begin(921600);
  WiFi.mode(WIFI_STA);
  WiFi.disconnect();
  esp_wifi_set_ps(WIFI_PS_NONE);
  esp_wifi_set_channel(CHANNEL, WIFI_SECOND_CHAN_NONE);

  esp_now_init();
  esp_now_peer_info_t peer = {};
  memcpy(peer.peer_addr, BCAST, 6);
  peer.channel = CHANNEL;
  peer.ifidx = WIFI_IF_STA;
  esp_now_add_peer(&peer);
  esp_now_register_recv_cb(on_recv);

  // send ESP-NOW as HT20 MCS0 so receivers get OFDM CSI (default 1 Mbps DSSS has none)
#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)
  esp_now_rate_config_t rc = {};
  rc.phymode = WIFI_PHY_MODE_HT20;
  rc.rate = WIFI_PHY_RATE_MCS0_LGI;
  esp_now_set_peer_rate_config(BCAST, &rc);
#else
  esp_wifi_config_espnow_rate(WIFI_IF_STA, WIFI_PHY_RATE_MCS0_LGI);
#endif

  wifi_csi_config_t cfg = {};
  cfg.lltf_en = true;
  cfg.htltf_en = false;
  cfg.stbc_htltf2_en = false;
  cfg.ltf_merge_en = false;
  cfg.channel_filter_en = false;
  cfg.manu_scale = false;
  esp_wifi_set_csi_config(&cfg);
  esp_wifi_set_csi_rx_cb(csi_cb, NULL);
  esp_wifi_set_csi(true);

  Serial.printf("I,hallsense,node=%d,ch=%d,mac=%s\n", NODE_ID, CHANNEL, WiFi.macAddress().c_str());
}

void loop() {
  static uint32_t next_tx = 0, next_rep = 1000;
  uint32_t now = millis();
  if ((int32_t)(now - next_tx) >= 0) {
    beacon_t bc = {MAGIC, T_BEACON, (uint8_t)NODE_ID, seq++};
    esp_now_send(BCAST, (uint8_t *)&bc, sizeof(bc));
    next_tx = now + 1000 / FS + (esp_random() % 11) - 5;    // jitter avoids lock-step collisions
  }
  if ((int32_t)(now - next_rep) >= 0) {
    flush_reports();
    next_rep += 1000;
  }
  delay(1);
}
