//! Small CPU-drawn pictures for the HUD: the CSI waterfall of a link and the drift sparkline.

use bevy::asset::RenderAssetUsages;
use bevy::image::ImageSampler;
use bevy::prelude::*;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::live_core::DriftSecond;

use super::palette::{RED_RGB, BACKGROUND_RGB, MUTED_RGB, PANEL_RGB, TEXT_RGB};

const WATERFALL_LOW_DB: f32 = -12.0;
const WATERFALL_HIGH_DB: f32 = 8.0;
pub const SPARK_WIDTH: u32 = 360;
pub const SPARK_HEIGHT: u32 = 84;
const BACKGROUND: [u8; 3] = BACKGROUND_RGB;
const SPARK_BACKGROUND: [u8; 3] = PANEL_RGB;

pub fn blank(width: u32, height: u32) -> Image {
    let mut image = Image::new_fill(
        Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[BACKGROUND[0], BACKGROUND[1], BACKGROUND[2], 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
    );
    image.sampler = ImageSampler::nearest();
    image
}

fn put(data: &mut [u8], width: u32, x: u32, y: u32, [r, g, b]: [u8; 3]) {
    let at = ((y * width + x) * 4) as usize;
    if let Some(pixel) = data.get_mut(at..at + 4) {
        pixel.copy_from_slice(&[r, g, b, 255]);
    }
}

/// Time left to right (newest at the right edge), subcarriers low to high frequency bottom to top,
/// each subcarrier's strength over the packet's mean as a colour from dark to hot.
pub fn paint_waterfall(image: &mut Image, amplitudes_db: &[Vec<f32>], packets: usize) {
    let subcarriers = amplitudes_db.last().map_or(1, Vec::len).max(1) as u32;
    let width = packets as u32;
    if image.width() != width || image.height() != subcarriers {
        *image = blank(width, subcarriers);
    }
    let Some(data) = image.data.as_mut() else { return };
    let start = width as usize - amplitudes_db.len().min(packets);
    for x in 0..width {
        let packet = (x as usize).checked_sub(start).and_then(|i| amplitudes_db.get(i));
        for k in 0..subcarriers {
            let color = match packet.and_then(|p| p.get(k as usize)) {
                Some(&db) => heat((db - WATERFALL_LOW_DB) / (WATERFALL_HIGH_DB - WATERFALL_LOW_DB)),
                None => BACKGROUND,
            };
            put(data, width, x, subcarriers - 1 - k, color);
        }
    }
}

/// Near-black to off-white; the power keeps an ordinary subcarrier a mid grey and the peaks bright.
fn heat(share: f32) -> [u8; 3] {
    let t = share.clamp(0.0, 1.0).powf(1.6);
    [0, 1, 2].map(|c| (f32::from(BACKGROUND[c]) + (f32::from(TEXT_RGB[c]) - f32::from(BACKGROUND[c])) * t) as u8)
}

/// One column per second, the line above which the room counts as moving drawn across.
pub fn paint_sparkline(image: &mut Image, history: &[DriftSecond], seconds_shown: usize) {
    if image.width() != SPARK_WIDTH || image.height() != SPARK_HEIGHT {
        *image = blank(SPARK_WIDTH, SPARK_HEIGHT);
    }
    let Some(data) = image.data.as_mut() else { return };
    let threshold = history.last().map_or(0.25, |s| s.moving_above);
    let peak = history.iter().map(|s| s.drift_db).fold(threshold * 2.2, f64::max);
    let row_of = |value: f64| {
        let share = (value / peak).clamp(0.0, 1.0);
        (SPARK_HEIGHT - 1) - ((SPARK_HEIGHT - 1) as f64 * share) as u32
    };
    let column_width = SPARK_WIDTH as f64 / seconds_shown as f64;
    let first = seconds_shown.saturating_sub(history.len());
    let (above, below) = (RED_RGB, MUTED_RGB.map(|c| c / 2));
    for x in 0..SPARK_WIDTH {
        let second = (f64::from(x) / column_width) as usize;
        let value = second.checked_sub(first).and_then(|i| history.get(i));
        let top = value.map_or(SPARK_HEIGHT, |s| row_of(s.drift_db));
        let color = if value.is_some_and(|s| s.drift_db > s.moving_above) { above } else { below };
        for y in 0..SPARK_HEIGHT {
            let fill = if y >= top { color } else { SPARK_BACKGROUND };
            put(data, SPARK_WIDTH, x, y, fill);
        }
    }
    let threshold_row = row_of(threshold);
    for x in (0..SPARK_WIDTH).filter(|x| x % 8 < 5) {
        put(data, SPARK_WIDTH, x, threshold_row, TEXT_RGB);
    }
}
