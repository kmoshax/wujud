use std::sync::Arc;

use eframe::egui::{self, Color32, CornerRadius, FontFamily, FontId, Margin, Stroke, TextStyle};

use crate::live_core::Source;

pub const BACKGROUND: Color32 = Color32::from_rgb(9, 9, 9);
pub const PANEL: Color32 = Color32::from_rgb(13, 13, 13);
pub const CARD: Color32 = Color32::from_rgb(17, 17, 17);
pub const CARD_EDGE: Color32 = Color32::from_rgb(40, 40, 39);
pub const TRACK: Color32 = Color32::from_rgb(32, 32, 31);
pub const TEXT: Color32 = Color32::from_rgb(232, 228, 220);
pub const MUTED: Color32 = Color32::from_rgb(122, 120, 116);
pub const FAINT: Color32 = Color32::from_rgb(58, 57, 55);
pub const RED: Color32 = Color32::from_rgb(229, 72, 77);
pub const ACCENT: Color32 = RED;
pub const EMPTY: Color32 = MUTED;
pub const STILL: Color32 = TEXT;
pub const MOVING: Color32 = RED;
pub const ALIVE: Color32 = TEXT;
pub const SILENT: Color32 = FAINT;

const PIXEL_FAMILY: &str = "pixel";
/// The pixel font's grid: it stays crisp at whole multiples of this.
const PIXEL: f32 = 8.0;

pub fn pixel(scale: f32) -> FontId {
    FontId::new(PIXEL * scale, FontFamily::Name(PIXEL_FAMILY.into()))
}

pub fn pixel_text(text: impl Into<String>, scale: f32, color: Color32) -> egui::RichText {
    egui::RichText::new(text).font(pixel(scale)).color(color)
}

fn load_pixel_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();
    fonts.font_data.insert(
        PIXEL_FAMILY.to_owned(),
        Arc::new(egui::FontData::from_static(include_bytes!("../assets/fonts/PressStart2P-Regular.ttf"))),
    );
    let fallbacks = fonts.families.get(&FontFamily::Proportional).cloned().unwrap_or_default();
    let pixel_then_fallbacks = std::iter::once(PIXEL_FAMILY.to_owned()).chain(fallbacks).collect();
    fonts.families.insert(FontFamily::Name(PIXEL_FAMILY.into()), pixel_then_fallbacks);
    ctx.set_fonts(fonts);
}

pub fn apply(ctx: &egui::Context) {
    load_pixel_font(ctx);
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL;
    visuals.window_fill = PANEL;
    visuals.extreme_bg_color = BACKGROUND;
    visuals.faint_bg_color = CARD;
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = RED.gamma_multiply(0.6);
    visuals.window_corner_radius = CornerRadius::same(1);
    visuals.menu_corner_radius = CornerRadius::same(1);
    visuals.window_stroke = Stroke::new(1.0_f32, CARD_EDGE);
    visuals.window_shadow = egui::Shadow::NONE;
    visuals.popup_shadow = egui::Shadow::NONE;
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, CARD_EDGE);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    for widget in [
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
    ] {
        widget.corner_radius = CornerRadius::same(1);
    }
    ctx.set_visuals(visuals);
    ctx.style_mut(|style| {
        style.spacing.item_spacing = egui::vec2(8.0, 6.0);
        style.spacing.window_margin = Margin::same(14);
        style.text_styles.insert(TextStyle::Body, FontId::proportional(14.0));
        style.text_styles.insert(TextStyle::Small, FontId::proportional(11.5));
    });
}

pub fn card(ui: &mut egui::Ui, add_contents: impl FnOnce(&mut egui::Ui)) {
    egui::Frame::new()
        .fill(CARD)
        .stroke(Stroke::new(1.0_f32, CARD_EDGE))
        .corner_radius(CornerRadius::same(1))
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add_contents(ui);
        });
    ui.add_space(10.0);
}

pub fn caption(ui: &mut egui::Ui, text: &str) {
    ui.label(pixel_text(text.to_uppercase(), 1.0, MUTED));
    ui.add_space(2.0);
}

/// A labelled bar: name on the left, value on the right in the pixel font, a thin flat fill underneath.
pub fn meter(ui: &mut egui::Ui, name: &str, fraction: f32, value: &str, color: Color32) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(name).color(TEXT));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(pixel_text(value, 1.0, color));
        });
    });
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 4.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::ZERO, TRACK);
    let mut filled = rect;
    filled.set_width(rect.width() * fraction.clamp(0.0, 1.0));
    if filled.width() > 0.5 {
        painter.rect_filled(filled, CornerRadius::ZERO, color);
    }
    ui.add_space(4.0);
}

/// Simulation outlined and muted; real data in off-white, live with a red dot.
pub fn source_tag(ui: &mut egui::Ui, source: Source) {
    let (edge, words) = match source {
        Source::Simulation => (MUTED, MUTED),
        Source::Live | Source::Replay => (TEXT, TEXT),
    };
    egui::Frame::new()
        .stroke(Stroke::new(1.0_f32, edge))
        .inner_margin(Margin::symmetric(10, 8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.horizontal(|ui| {
                if source == Source::Live {
                    status_dot(ui, RED, 4.0, PIXEL * 1.5);
                }
                ui.label(pixel_text(source.tag(), 1.5, words));
            });
            ui.label(egui::RichText::new(source.note()).small().color(MUTED));
        });
    ui.add_space(10.0);
}

pub fn status_dot(ui: &mut egui::Ui, color: Color32, radius: f32, row_height: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(radius * 2.0, row_height), egui::Sense::hover());
    let square = egui::Rect::from_center_size(rect.center(), egui::vec2(radius * 1.6, radius * 1.6));
    ui.painter().rect_filled(square, CornerRadius::ZERO, color);
}
