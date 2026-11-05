//! Near-black, greys and a warm off-white; one clean red, only for what moves.

use bevy::color::Mix;
use bevy::prelude::*;

pub const BACKGROUND: Color = Color::srgb_u8(9, 9, 9);
pub const FLOOR: Color = Color::srgb_u8(14, 14, 14);
pub const GRID: Color = Color::srgb_u8(30, 30, 30);
pub const EDGE: Color = Color::srgb_u8(72, 72, 70);
pub const FAINT: Color = Color::srgb_u8(58, 57, 55);
pub const MUTED: Color = Color::srgb_u8(122, 120, 116);
pub const TEXT: Color = Color::srgb_u8(232, 228, 220);
pub const RED: Color = Color::srgb_u8(229, 72, 77);
/// The ESP32 DevKit's blue LED, lit.
pub const LED_BLUE: Color = Color::srgb_u8(60, 140, 255);

pub const BACKGROUND_RGB: [u8; 3] = [9, 9, 9];
pub const PANEL_RGB: [u8; 3] = [16, 16, 16];
pub const MUTED_RGB: [u8; 3] = [122, 120, 116];
pub const TEXT_RGB: [u8; 3] = [232, 228, 220];
pub const RED_RGB: [u8; 3] = [229, 72, 77];

/// Brighter than white in linear light, so bloom lets it glow a little.
pub fn glowing(color: Color, strength: f32) -> LinearRgba {
    let linear = color.to_linear();
    LinearRgba::new(linear.red * strength, linear.green * strength, linear.blue * strength, linear.alpha)
}

pub fn mix(from: Color, to: Color, share: f32) -> Color {
    let (a, b) = (from.to_linear(), to.to_linear());
    Color::LinearRgba(a.mix(&b, share.clamp(0.0, 1.0)))
}
