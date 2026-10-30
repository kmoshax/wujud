/* Replay real CSI recordings through RuView's UNMODIFIED edge DSP (firmware/esp32-csi-node/main/edge_processing.c),
 * the same way RuView's own host tests do (test/test_edge_stbc.c: #include the .c, call process_frame()).
 * Config = the firmware's boot defaults (nvs_config.c): tier 2, presence_thresh 0 (auto-calibrate), top_k 8, fall 15.
 * stdin : repeated frames  u64 timestamp_us | i8 rssi | u8 channel | u16 iq_len | iq_len bytes   (little endian)
 * stdout: one line per frame  motion_energy presence_flag calibrated adaptive_threshold
 * Build: see build.sh */
#define _USE_MATH_DEFINES
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define TickType_t unsigned
#define portNUM_PROCESSORS 2
#include "edge_processing.c"
#include <stdio.h>

nvs_config_t g_nvs_config;
uint8_t csi_collector_get_node_id(void) { return 1; }

int main(void)
{
    edge_config_t cfg = {.tier = 2, .presence_thresh = 0.0f, .fall_thresh = 15.0f,
                         .vital_interval_ms = 1000, .top_k_count = 8, .power_duty = 100};
    if (edge_processing_init(&cfg) != ESP_OK) return 1;
    edge_ring_slot_t slot;
    uint64_t ts; int8_t rssi; uint8_t ch; uint16_t len;
    while (fread(&ts, 8, 1, stdin) == 1 && fread(&rssi, 1, 1, stdin) == 1 && fread(&ch, 1, 1, stdin) == 1 &&
           fread(&len, 2, 1, stdin) == 1) {
        memset(&slot, 0, sizeof(slot));
        if (len > EDGE_MAX_IQ_BYTES || fread(slot.iq_data, 1, len, stdin) != len) return 2;
        slot.iq_len = len; slot.rssi = rssi; slot.channel = ch; slot.timestamp_us = (uint32_t)ts;
        process_frame(&slot);
        printf("%.6g %d %d %.6g\n", s_motion_energy, s_presence_detected ? 1 : 0, s_calibrated ? 1 : 0, s_adaptive_threshold);
    }
    return 0;
}
