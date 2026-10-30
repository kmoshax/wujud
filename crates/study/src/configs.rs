//! The hardware configurations compared by the study and what each costs on top of the baseline.

use serde::Serialize;
use wujud_sim::band::{
    channel_2g4_hz, channel_5g_hz, BandPlan, FrequencyEffects, SubcarrierLayout,
};

/// One ESP32 board (2.4 GHz, e.g. ESP32-S3/C3 dev board), the baseline receiver (USD, 2026 retail).
pub const ESP32_BOARD_USD: f64 = 5.0;
/// ESP32-C6 (Wi-Fi 6, 2.4 GHz only, HE20 CSI).
pub const ESP32_C6_BOARD_USD: f64 = 6.0;
/// ESP32-C5 (dual-band Wi-Fi 6, the only ESP32 with 5 GHz).
pub const ESP32_C5_BOARD_USD: f64 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Factor {
    Baseline,
    Band,
    Bandwidth,
    Hopping,
    PacketRate,
    Links,
    Placement,
    Combination,
}

#[derive(Clone, Debug, Serialize)]
pub struct StudyConfig {
    pub name: String,
    pub factor: Factor,
    pub description: String,
    pub band: BandPlan,
    pub packets_per_second: f64,
    pub boards: usize,
    pub mesh: bool,
    pub smart_placement: bool,
    /// The measured link runs ESP32 -> ESP32 (a router cannot hop channels).
    pub dedicated_transmitter: bool,
    pub extra_cost_usd: f64,
    pub cost_note: String,
}

impl StudyConfig {
    fn baseline() -> Self {
        StudyConfig {
            name: "baseline".into(),
            factor: Factor::Baseline,
            description:
                "2.4 GHz ch 6, HT20 L-LTF 52, 50 pkt/s, router -> 1 ESP32, random placement".into(),
            band: BandPlan::single(channel_2g4_hz(6), SubcarrierLayout::LegacyLtf52),
            packets_per_second: 50.0,
            boards: 1,
            mesh: false,
            smart_placement: false,
            dedicated_transmitter: false,
            extra_cost_usd: 0.0,
            cost_note: "1 ESP32 + existing router".into(),
        }
    }

    fn variant(
        name: &str,
        factor: Factor,
        description: &str,
        extra_cost_usd: f64,
        cost_note: &str,
    ) -> Self {
        StudyConfig {
            name: name.into(),
            factor,
            description: description.into(),
            extra_cost_usd,
            cost_note: cost_note.into(),
            ..StudyConfig::baseline()
        }
    }

    fn on(mut self, band: BandPlan) -> Self {
        self.band = band;
        self
    }

    fn rate(mut self, packets_per_second: f64) -> Self {
        self.packets_per_second = packets_per_second;
        self
    }

    fn boards(mut self, boards: usize, mesh: bool) -> Self {
        self.boards = boards;
        self.mesh = mesh;
        self
    }

    fn smart(mut self) -> Self {
        self.smart_placement = true;
        self
    }

    fn dedicated_transmitter(mut self) -> Self {
        self.dedicated_transmitter = true;
        self
    }

    pub fn links(&self) -> usize {
        let mesh_links = if self.mesh {
            self.boards * (self.boards - 1) / 2
        } else {
            0
        };
        self.boards + mesh_links
    }
}

/// 5 GHz with every uncertain effect set against detection (see README "Physics assumptions").
pub fn five_ghz_conservative() -> FrequencyEffects {
    FrequencyEffects {
        body_absorption_range: None,
        person_absorption_m2: 0.4,
        tx_power_offset_db: -3.0,
        evm_offset_db: 2.0,
    }
}

/// 5 GHz with the body more opaque and absorbing and the radio as good as at 2.4 GHz.
pub fn five_ghz_optimistic() -> FrequencyEffects {
    FrequencyEffects {
        body_absorption_range: Some((0.9, 0.98)),
        person_absorption_m2: 0.55,
        tx_power_offset_db: 0.0,
        evm_offset_db: 0.0,
    }
}

fn five_ghz(layout: SubcarrierLayout, effects: FrequencyEffects) -> BandPlan {
    let carrier = match layout {
        SubcarrierLayout::Ht40Ltf114 => channel_5g_hz(38),
        _ => channel_5g_hz(36),
    };
    BandPlan::single(carrier, layout).with_effects(effects)
}

fn two_ghz(layout: SubcarrierLayout) -> BandPlan {
    let carrier = match layout {
        SubcarrierLayout::Ht40Ltf114 => channel_2g4_hz(8),
        _ => channel_2g4_hz(6),
    };
    BandPlan::single(carrier, layout)
}

pub fn all_configs() -> Vec<StudyConfig> {
    use Factor::*;
    use SubcarrierLayout::*;
    let c5 = ESP32_C5_BOARD_USD - ESP32_BOARD_USD;
    let conservative = five_ghz_conservative();
    let optimistic = five_ghz_optimistic();
    vec![
        StudyConfig::baseline(),
        StudyConfig::variant(
            "5ghz_conservative",
            Band,
            "5 GHz ch 36, conservative body/radio effects",
            c5,
            "ESP32-C5 instead of ESP32; dual-band router",
        )
        .on(five_ghz(LegacyLtf52, conservative)),
        StudyConfig::variant(
            "5ghz_optimistic",
            Band,
            "5 GHz ch 36, optimistic body/radio effects",
            c5,
            "ESP32-C5 instead of ESP32; dual-band router",
        )
        .on(five_ghz(LegacyLtf52, optimistic)),
        StudyConfig::variant(
            "ht40_114",
            Bandwidth,
            "2.4 GHz HT40 (ch 6+10), 114 subcarriers",
            0.0,
            "router must run 40 MHz at 2.4 GHz",
        )
        .on(two_ghz(Ht40Ltf114)),
        StudyConfig::variant(
            "he20_242",
            Bandwidth,
            "2.4 GHz HE20 (Wi-Fi 6), 242 subcarriers",
            ESP32_C6_BOARD_USD - ESP32_BOARD_USD,
            "ESP32-C6; Wi-Fi 6 router",
        )
        .on(two_ghz(He20Ltf242)),
        StudyConfig::variant(
            "hop3_2g4",
            Hopping,
            "ESP32 -> ESP32 hopping ch 1/6/11, 50 pkt/s total",
            ESP32_BOARD_USD,
            "second ESP32 as transmitter",
        )
        .on(BandPlan::hopping(
            vec![channel_2g4_hz(1), channel_2g4_hz(6), channel_2g4_hz(11)],
            LegacyLtf52,
        ))
        .dedicated_transmitter(),
        StudyConfig::variant(
            "hop4_5g",
            Hopping,
            "ESP32-C5 pair hopping ch 36/64/100/149 (conservative), 50 pkt/s total",
            2.0 * ESP32_C5_BOARD_USD - ESP32_BOARD_USD,
            "two ESP32-C5 (transmitter + receiver)",
        )
        .on(
            BandPlan::hopping([36, 64, 100, 149].map(channel_5g_hz).to_vec(), LegacyLtf52)
                .with_effects(conservative),
        )
        .dedicated_transmitter(),
        StudyConfig::variant("rate25", PacketRate, "25 pkt/s", 0.0, "firmware only").rate(25.0),
        StudyConfig::variant(
            "rate100",
            PacketRate,
            "100 pkt/s",
            0.0,
            "firmware only; more airtime",
        )
        .rate(100.0),
        StudyConfig::variant(
            "rate200",
            PacketRate,
            "200 pkt/s",
            0.0,
            "firmware only; more airtime",
        )
        .rate(200.0),
        StudyConfig::variant(
            "links2",
            Links,
            "router -> 2 ESP32",
            ESP32_BOARD_USD,
            "+1 ESP32",
        )
        .boards(2, false),
        StudyConfig::variant(
            "links3",
            Links,
            "router -> 3 ESP32",
            2.0 * ESP32_BOARD_USD,
            "+2 ESP32",
        )
        .boards(3, false),
        StudyConfig::variant(
            "mesh2",
            Links,
            "router -> 2 ESP32 + ESP32 <-> ESP32 (3 links)",
            ESP32_BOARD_USD,
            "+1 ESP32",
        )
        .boards(2, true),
        StudyConfig::variant(
            "smart",
            Placement,
            "router and board on opposite walls, link crosses the bed",
            0.0,
            "installation only",
        )
        .smart(),
        StudyConfig::variant(
            "smart_rate100",
            Combination,
            "smart placement, 100 pkt/s",
            0.0,
            "installation + firmware",
        )
        .smart()
        .rate(100.0),
        StudyConfig::variant(
            "smart_mesh2",
            Combination,
            "smart placement, 2 ESP32 mesh (3 links)",
            ESP32_BOARD_USD,
            "+1 ESP32",
        )
        .smart()
        .boards(2, true),
        StudyConfig::variant(
            "5ghz_smart",
            Combination,
            "5 GHz (conservative), smart placement",
            c5,
            "ESP32-C5",
        )
        .on(five_ghz(LegacyLtf52, conservative))
        .smart(),
        StudyConfig::variant(
            "5ghz_he20",
            Combination,
            "5 GHz HE20 242 (conservative)",
            c5,
            "ESP32-C5; Wi-Fi 6 router",
        )
        .on(five_ghz(He20Ltf242, conservative)),
        StudyConfig::variant(
            "5ghz_ht40_rate100",
            Combination,
            "5 GHz HT40 114 (conservative), 100 pkt/s",
            c5,
            "ESP32-C5",
        )
        .on(five_ghz(Ht40Ltf114, conservative))
        .rate(100.0),
        StudyConfig::variant(
            "best_2g4",
            Combination,
            "2.4 GHz HT40, 100 pkt/s, smart, 2 ESP32 mesh",
            ESP32_BOARD_USD,
            "+1 ESP32; router at 40 MHz",
        )
        .on(two_ghz(Ht40Ltf114))
        .rate(100.0)
        .smart()
        .boards(2, true),
        StudyConfig::variant(
            "best_conservative",
            Combination,
            "5 GHz HT40 (conservative), 100 pkt/s, smart, 2 ESP32 mesh",
            2.0 * ESP32_C5_BOARD_USD - ESP32_BOARD_USD,
            "two ESP32-C5",
        )
        .on(five_ghz(Ht40Ltf114, conservative))
        .rate(100.0)
        .smart()
        .boards(2, true),
        StudyConfig::variant(
            "best_optimistic",
            Combination,
            "5 GHz HT40 (optimistic), 100 pkt/s, smart, 2 ESP32 mesh",
            2.0 * ESP32_C5_BOARD_USD - ESP32_BOARD_USD,
            "two ESP32-C5",
        )
        .on(five_ghz(Ht40Ltf114, optimistic))
        .rate(100.0)
        .smart()
        .boards(2, true),
    ]
}
