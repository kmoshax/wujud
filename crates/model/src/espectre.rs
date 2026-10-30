//! Reader for the ESPectre real ESP32 CSI dataset (`dataset_info.json` + one `.npz` per recording).

use std::collections::BTreeMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use ndarray::{Array1, Array2};
use ndarray_npy::NpzReader;
use serde::Deserialize;

use crate::numeric::unwrap_u32_microseconds;
use crate::{ModelError, Result};

#[derive(Debug, Clone, Deserialize)]
pub struct RecordingInfo {
    pub filename: String,
    pub chip: String,
    pub subcarriers: usize,
    pub collected_at: String,
    pub environment: Option<String>,
    pub num_packets: Option<usize>,
    /// ESPectre's split: `train`, `selection`, `holdout` or `exclude`.
    pub dataset_role: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
struct DatasetInfo {
    files: BTreeMap<String, Vec<RecordingInfo>>,
}

/// One recording, packets sorted by receive time; `times` start at 0 s, `iq` is `[packets, 2 * subcarriers]`.
#[derive(Debug, Clone)]
pub struct Recording {
    pub times: Vec<f64>,
    pub iq: Array2<i8>,
    pub rssi: Vec<f64>,
}

impl Recording {
    /// The packets received before `seconds`.
    pub fn first_seconds(&self, seconds: f64) -> Recording {
        let keep = self.times.iter().take_while(|&&t| t < seconds).count();
        Recording {
            times: self.times[..keep].to_vec(),
            iq: self.iq.slice(ndarray::s![..keep, ..]).to_owned(),
            rssi: self.rssi[..keep].to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EspectreDataset {
    pub root: PathBuf,
    /// Label folder (`empty`, `motion`, `static_presence`) -> recordings.
    pub files: BTreeMap<String, Vec<RecordingInfo>>,
}

impl EspectreDataset {
    pub fn open(root: &Path) -> Result<Self> {
        let path = root.join("dataset_info.json");
        let text = std::fs::read_to_string(&path).map_err(|source| ModelError::Io {
            path: path.clone(),
            source,
        })?;
        let info: DatasetInfo =
            serde_json::from_str(&text).map_err(|source| ModelError::Json { path, source })?;
        Ok(EspectreDataset {
            root: root.into(),
            files: info.files,
        })
    }

    /// The first recording of `label` whose `collected_at` starts with `collected_at_prefix`.
    pub fn find(
        &self,
        label: &str,
        collected_at_prefix: &str,
        environment: Option<&str>,
    ) -> Option<&RecordingInfo> {
        self.files.get(label)?.iter().find(|info| {
            info.collected_at.starts_with(collected_at_prefix)
                && environment.is_none_or(|env| info.environment.as_deref() == Some(env))
        })
    }

    pub fn load(&self, label: &str, info: &RecordingInfo) -> Result<Recording> {
        load_recording(&self.root.join(label).join(&info.filename))
    }
}

fn npz_error(path: &Path, name: &str, error: impl std::fmt::Display) -> ModelError {
    ModelError::Npz {
        path: path.into(),
        name: name.into(),
        message: error.to_string(),
    }
}

/// Loads one `.npz`: unwraps the 32-bit microsecond clock and sorts packets by time (stable).
pub fn load_recording(path: &Path) -> Result<Recording> {
    let file = File::open(path).map_err(|source| ModelError::Io {
        path: path.into(),
        source,
    })?;
    let mut npz = NpzReader::new(file).map_err(|e| npz_error(path, "", e))?;
    let timestamps: Array1<u32> = npz
        .by_name("wifi_rx_ts_us")
        .map_err(|e| npz_error(path, "wifi_rx_ts_us", e))?;
    let rssi: Array1<i16> = npz
        .by_name("rssi_dbm")
        .map_err(|e| npz_error(path, "rssi_dbm", e))?;
    let csi: Array2<i8> = npz
        .by_name("csi_data")
        .map_err(|e| npz_error(path, "csi_data", e))?;
    let seconds: Vec<f64> = unwrap_u32_microseconds(timestamps.as_slice().unwrap())
        .iter()
        .map(|us| us / 1e6)
        .collect();
    let mut order: Vec<usize> = (0..seconds.len()).collect();
    order.sort_by(|&a, &b| seconds[a].total_cmp(&seconds[b]));
    let start = order.first().map_or(0.0, |&i| seconds[i]);
    Ok(Recording {
        times: order.iter().map(|&i| seconds[i] - start).collect(),
        iq: csi.select(ndarray::Axis(0), &order),
        rssi: order.iter().map(|&i| rssi[i] as f64).collect(),
    })
}

/// Class index of an ESPectre label folder in the room benchmarks (0 empty, 1 still person, 2 motion).
pub fn room_class(label: &str) -> Option<i64> {
    match label {
        "empty" => Some(0),
        "static_presence" => Some(1),
        "motion" => Some(2),
        _ => None,
    }
}
