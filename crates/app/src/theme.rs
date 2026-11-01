use eframe::egui::{self, Color32, CornerRadius, FontId, Margin, Stroke, TextStyle};

pub const BACKGROUND: Color32 = Color32::from_rgb(11, 14, 20);
pub const PANEL: Color32 = Color32::from_rgb(16, 20, 28);
pub const CARD: Color32 = Color32::from_rgb(23, 28, 39);
pub const CARD_EDGE: Color32 = Color32::from_rgb(36, 43, 58);
pub const TRACK: Color32 = Color32::from_rgb(34, 40, 54);
pub const TEXT: Color32 = Color32::from_rgb(226, 232, 242);
pub const MUTED: Color32 = Color32::from_rgb(128, 140, 162);
pub const ACCENT: Color32 = Color32::from_rgb(64, 196, 255);
pub const EMPTY: Color32 = Color32::from_rgb(120, 134, 158);
pub const STILL: Color32 = Color32::from_rgb(72, 220, 170);
pub const MOVING: Color32 = Color32::from_rgb(255, 176, 64);
pub const ALIVE: Color32 = Color32::from_rgb(72, 220, 140);
pub const SILENT: Color32 = Color32::from_rgb(240, 90, 90);

pub fn apply(ctx: &egui::Context) {
    let mut visuals = egui::Visuals::dark();
    visuals.panel_fill = PANEL;
    visuals.window_fill = PANEL;
    visuals.extreme_bg_color = BACKGROUND;
    visuals.faint_bg_color = CARD;
    visuals.override_text_color = Some(TEXT);
    visuals.selection.bg_fill = ACCENT.gamma_multiply(0.6);
    visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, CARD_EDGE);
    visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0_f32, TEXT);
    for widget in [
        &mut visuals.widgets.inactive,
        &mut visuals.widgets.hovered,
        &mut visuals.widgets.active,
    ] {
        widget.corner_radius = CornerRadius::same(6);
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
        .corner_radius(CornerRadius::same(10))
        .inner_margin(Margin::same(12))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());
            add_contents(ui);
        });
    ui.add_space(10.0);
}

pub fn caption(ui: &mut egui::Ui, text: &str) {
    ui.label(
        egui::RichText::new(text.to_uppercase())
            .size(10.5)
            .color(MUTED)
            .extra_letter_spacing(1.2),
    );
}

/// A labelled bar: name on the left, value text on the right, a thin rounded fill underneath.
pub fn meter(ui: &mut egui::Ui, name: &str, fraction: f32, value: &str, color: Color32) {
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(name).color(TEXT));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.label(egui::RichText::new(value).color(color).strong().monospace());
        });
    });
    let (rect, _) = ui.allocate_exact_size(
        egui::vec2(ui.available_width(), 6.0),
        egui::Sense::hover(),
    );
    let painter = ui.painter();
    painter.rect_filled(rect, CornerRadius::same(3), TRACK);
    let mut filled = rect;
    filled.set_width(rect.width() * fraction.clamp(0.0, 1.0));
    if filled.width() > 0.5 {
        painter.rect_filled(filled, CornerRadius::same(3), color);
    }
    ui.add_space(4.0);
}

pub fn status_dot(ui: &mut egui::Ui, color: Color32, radius: f32, row_height: f32) {
    let (rect, _) = ui.allocate_exact_size(egui::vec2(radius * 2.0, row_height), egui::Sense::hover());
    ui.painter().circle_filled(rect.center(), radius * 1.8, color.gamma_multiply(0.18));
    ui.painter().circle_filled(rect.center(), radius, color);
}
