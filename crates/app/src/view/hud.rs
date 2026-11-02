//! The words on screen, in a pixel font at whole multiples of its 8 px grid: the room's state,
//! breathing, where the person is, the drift the motion detector reads against its threshold, and
//! each link's raw CSI as a waterfall.

use bevy::prelude::*;
use bevy::text::FontSmoothing;
use bevy::ui::UiSystem;
use bevy::window::PrimaryWindow;

use crate::live_core::{RoomStatus, Snapshot};
use crate::skeleton::Gait;

use super::palette::{RED, EDGE, FAINT, GRID, MUTED, TEXT};
use super::person::{Figure, BREATHING_AFTER_STILL_SECONDS};
use super::raster::{self, SPARK_HEIGHT, SPARK_WIDTH};
use super::{world, Latest, WATERFALL_PACKETS};

const MARGIN: f32 = 32.0;
const STRIP_WIDTH: f32 = 400.0;
/// Narrower than this, the signal strip would cover the room: it hides until w shows it.
const ROOM_FOR_STRIP: f32 = 1400.0;
const DRIFT_SECONDS_SHOWN: usize = 120;
const LINK_BLOCK_CHROME: f32 = 40.0;
const STRIP_CHROME: f32 = 140.0;
/// The font's pixel grid: text is drawn at whole multiples of it.
const PIXEL: f32 = 8.0;
const MOST_DISTURBED_ABOVE: f32 = 0.12;

pub struct HudPlugin;

impl Plugin for HudPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Strip>().add_systems(Startup, (load_font, spawn_panels).chain()).add_systems(
            PostUpdate,
            (rebuild_strip, fit_strip, update_panels, update_strip, place_labels, count_frames)
                .chain()
                .before(UiSystem::Layout),
        );
    }
}

#[derive(Resource)]
struct PixelFont(Handle<Font>);

#[derive(Resource, Default)]
struct Strip {
    links: Vec<String>,
    boards: Vec<String>,
    window_height: f32,
    waterfalls: Vec<Handle<Image>>,
    /// Set by w: shown or hidden whatever the window's width.
    chosen: Option<bool>,
}

#[derive(Component)]
struct StripRoot;

fn fit_strip(
    keys: Res<ButtonInput<KeyCode>>,
    windows: Query<&Window>,
    mut strip: ResMut<Strip>,
    mut roots: Query<&mut Visibility, With<StripRoot>>,
) {
    let wide = windows.iter().next().is_some_and(|w| w.width() >= ROOM_FOR_STRIP);
    if keys.just_pressed(KeyCode::KeyW) {
        strip.chosen = Some(!strip.chosen.unwrap_or(wide));
    }
    let shown = strip.chosen.unwrap_or(wide);
    for mut visibility in &mut roots {
        *visibility = if shown { Visibility::Inherited } else { Visibility::Hidden };
    }
}

#[derive(Component)]
struct LabelRoot;

#[derive(Component)]
enum Readout {
    Source,
    State,
    Detail,
    Breath,
    Where,
    Drift,
    Frames,
}

#[derive(Component)]
struct Sparkline(Handle<Image>);

#[derive(Component)]
struct LinkRate(usize);

#[derive(Component)]
struct LinkBar(usize);

#[derive(Component)]
enum WorldLabel {
    Board(usize),
    Person,
}

fn load_font(mut commands: Commands, mut fonts: ResMut<Assets<Font>>) {
    let bytes = include_bytes!("../../assets/fonts/PressStart2P-Regular.ttf");
    let font = Font::try_from_bytes(bytes.to_vec()).expect("bundled font");
    commands.insert_resource(PixelFont(fonts.add(font)));
}

/// `scale` whole steps of the font's 8 px grid.
fn text(font: &PixelFont, value: &str, scale: u8, color: Color) -> (Text, TextFont, TextColor) {
    (
        Text::new(value),
        TextFont { font: font.0.clone(), font_size: PIXEL * f32::from(scale), font_smoothing: FontSmoothing::None, ..default() },
        TextColor(color),
    )
}

fn gap(pixels: f32) -> Node {
    Node { margin: UiRect::top(Val::Px(pixels)), ..default() }
}

fn spawn_panels(mut commands: Commands, font: Res<PixelFont>, latest: Res<Latest>, mut images: ResMut<Assets<Image>>) {
    let font = &*font;
    let spark = images.add(raster::blank(SPARK_WIDTH, SPARK_HEIGHT));
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: Val::Px(MARGIN),
            top: Val::Px(MARGIN),
            width: Val::Px(440.0),
            flex_direction: FlexDirection::Column,
            ..default()
        })
        .with_children(|panel| {
            panel.spawn(text(font, "WUJUD", 2, TEXT));
            panel.spawn((Readout::Source, text(font, "", 1, MUTED), gap(PIXEL)));
            panel.spawn((Readout::State, text(font, "", 6, MUTED), gap(5.0 * PIXEL)));
            panel.spawn((Readout::Detail, text(font, "", 2, TEXT), gap(2.0 * PIXEL)));
            panel.spawn((Readout::Breath, text(font, "", 2, TEXT), gap(2.0 * PIXEL)));
            panel.spawn((Readout::Where, text(font, "", 1, MUTED), gap(2.0 * PIXEL)));
            panel.spawn((text(font, "MOTION, DB/S", 1, MUTED), gap(5.0 * PIXEL)));
            panel.spawn((
                Sparkline(spark.clone()),
                ImageNode::new(spark),
                Node { width: Val::Px(SPARK_WIDTH as f32), height: Val::Px(SPARK_HEIGHT as f32), margin: UiRect::top(Val::Px(PIXEL)), ..default() },
            ));
            panel.spawn((Readout::Drift, text(font, "", 1, MUTED), gap(PIXEL)));
        });
    let mut legend = vec![
        "LINE: A WI-FI LINK. THICKER = A BODY DISTURBS IT",
        "RED ELLIPSE: ITS FRESNEL ZONE. THE PERSON IS",
        "WHERE THE LIT ZONES CROSS",
    ];
    if latest.0.simulated {
        legend.push("+ : WHERE THE SIMULATED PERSON REALLY IS");
    }
    commands
        .spawn(Node {
            position_type: PositionType::Absolute,
            left: Val::Px(MARGIN),
            bottom: Val::Px(MARGIN),
            flex_direction: FlexDirection::Column,
            row_gap: Val::Px(PIXEL),
            ..default()
        })
        .with_children(|column| {
            for line in legend {
                column.spawn(text(font, line, 1, MUTED));
            }
            column.spawn((text(font, "DRAG ORBIT  SCROLL ZOOM  SPACE AUTO  1-3 VIEWS  4 FOLLOW  W SIGNAL  F FULL  Q QUIT", 1, FAINT), gap(PIXEL)));
        });
    commands.spawn((
        Readout::Frames,
        text(font, "", 1, FAINT),
        Node { position_type: PositionType::Absolute, right: Val::Px(MARGIN), bottom: Val::Px(MARGIN), ..default() },
    ));
}

#[allow(clippy::type_complexity)]
fn rebuild_strip(
    mut commands: Commands,
    font: Res<PixelFont>,
    latest: Res<Latest>,
    mut strip: ResMut<Strip>,
    mut images: ResMut<Assets<Image>>,
    windows: Query<&Window, With<PrimaryWindow>>,
    roots: Query<Entity, Or<(With<StripRoot>, With<LabelRoot>)>>,
) {
    let snapshot = &latest.0;
    let height = windows.single().map_or(900.0, |w| w.height());
    let links: Vec<String> = snapshot.links.iter().map(|l| l.name.clone()).collect();
    let boards: Vec<String> = snapshot.boards.iter().map(|b| b.name.clone()).collect();
    let same = strip.links == links && strip.boards == boards && (strip.window_height - height).abs() < 1.0;
    if same && !roots.is_empty() {
        return;
    }
    for root in &roots {
        commands.entity(root).despawn();
    }
    let font = &*font;
    let count = links.len().max(1) as f32;
    let waterfall_height = ((height - STRIP_CHROME) / count - LINK_BLOCK_CHROME).clamp(24.0, 160.0).floor();
    let waterfalls: Vec<Handle<Image>> = links.iter().map(|_| images.add(raster::blank(WATERFALL_PACKETS as u32, 1))).collect();
    commands
        .spawn((
            StripRoot,
            Node {
                position_type: PositionType::Absolute,
                right: Val::Px(MARGIN),
                top: Val::Px(MARGIN),
                width: Val::Px(STRIP_WIDTH),
                flex_direction: FlexDirection::Column,
                ..default()
            },
        ))
        .with_children(|column| {
            column.spawn(text(font, "RAW SIGNAL, WHAT THE NETWORKS READ", 1, MUTED));
            for (index, name) in links.iter().enumerate() {
                column
                    .spawn(Node { justify_content: JustifyContent::SpaceBetween, margin: UiRect::top(Val::Px(3.0 * PIXEL)), ..default() })
                    .with_children(|row| {
                        row.spawn(text(font, &name.to_uppercase(), 1, TEXT));
                        row.spawn((LinkRate(index), text(font, "", 1, MUTED)));
                    });
                column
                    .spawn((
                        Node { width: Val::Percent(100.0), height: Val::Px(2.0), margin: UiRect::vertical(Val::Px(PIXEL)), ..default() },
                        BackgroundColor(GRID),
                    ))
                    .with_child((LinkBar(index), Node { width: Val::Percent(0.0), height: Val::Percent(100.0), ..default() }, BackgroundColor(MUTED)));
                column.spawn((
                    ImageNode::new(waterfalls[index].clone()),
                    Node { width: Val::Percent(100.0), height: Val::Px(waterfall_height), ..default() },
                ));
            }
            column.spawn((text(font, "UP: SUBCARRIER  RIGHT: TIME (3 S)", 1, MUTED), gap(2.0 * PIXEL)));
            column.spawn((text(font, "BANDS THAT WOBBLE: SOMEONE MOVING", 1, MUTED), gap(PIXEL)));
        });
    commands
        .spawn((LabelRoot, Node { position_type: PositionType::Absolute, ..default() }))
        .with_children(|labels| {
            for (index, name) in boards.iter().enumerate() {
                labels.spawn((
                    WorldLabel::Board(index),
                    text(font, &name.to_uppercase(), 1, MUTED),
                    TextLayout::new_with_no_wrap(),
                    Node { position_type: PositionType::Absolute, ..default() },
                ));
            }
            labels.spawn((
                WorldLabel::Person,
                text(font, "", 1, TEXT),
                TextLayout::new_with_no_wrap(),
                Node { position_type: PositionType::Absolute, ..default() },
            ));
        });
    *strip = Strip { links, boards, window_height: height, waterfalls, chosen: strip.chosen };
}

fn set(text: &mut Text, value: String) {
    if text.0 != value {
        text.0 = value;
    }
}

fn ago(seconds: f64) -> String {
    if seconds < 120.0 {
        format!("{seconds:.0} s ago")
    } else {
        format!("{:.0} min ago", seconds / 60.0)
    }
}

fn state_words(snapshot: &Snapshot) -> (&'static str, String, Color) {
    match snapshot.status {
        RoomStatus::Learning => ("LEARNING", "first answer in 20 s".into(), MUTED),
        RoomStatus::Waiting => ("WAITING", "for the first movement".into(), MUTED),
        RoomStatus::Empty { quiet_for: Some(seconds) } => ("QUIET", format!("no movement for {:.0} min", seconds / 60.0), MUTED),
        RoomStatus::Empty { quiet_for: None } => ("EMPTY", "nobody here".into(), MUTED),
        RoomStatus::Still { last_moved: Some(seconds) } if seconds < 1.0 => ("STILL", "small movements".into(), TEXT),
        RoomStatus::Still { last_moved: Some(seconds) } => ("STILL", format!("last moved {}", ago(seconds)), TEXT),
        RoomStatus::Still { last_moved: None } => ("STILL", format!("still for {:.0} s", snapshot.still_seconds), TEXT),
        RoomStatus::Moving => ("MOVING", "someone is walking".into(), RED),
    }
}

fn breath_words(snapshot: &Snapshot) -> String {
    match (snapshot.breath_bpm, snapshot.status) {
        (Some(bpm), _) => format!("breathing {bpm:.0}/min"),
        (None, RoomStatus::Still { .. }) => {
            let left = BREATHING_AFTER_STILL_SECONDS - snapshot.still_seconds;
            if left > 0.0 { format!("breath in {left:.0} s") } else { "breath too faint".into() }
        }
        _ => String::new(),
    }
}

fn where_words(snapshot: &Snapshot, figure: &Figure) -> String {
    if snapshot.person.is_none() {
        return String::new();
    }
    if snapshot.one_line {
        return "TWO BOARDS: HOW NEAR THEIR LINE, NOT WHERE".into();
    }
    if !snapshot.placed {
        return "BOARDS NOT PLACED: NO SPOT SHOWN".into();
    }
    let Some(at) = figure.at.filter(|_| figure.located) else {
        return String::new();
    };
    format!("AT X {:.1} M  Y {:.1} M", at[0], at[1])
}

fn source_words(snapshot: &Snapshot) -> String {
    let source = if snapshot.simulated { "SIMULATOR" } else { "LIVE" };
    let rate: f64 = snapshot.links.iter().map(|l| l.packets_per_second).sum();
    format!("{source}  {} BOARDS  {} LINKS  {rate:.0} PKT/S", snapshot.boards.len(), snapshot.links.len())
}

fn update_panels(
    latest: Res<Latest>,
    figure: Res<Figure>,
    mut readouts: Query<(&Readout, &mut Text, &mut TextColor)>,
    sparklines: Query<&Sparkline>,
    mut images: ResMut<Assets<Image>>,
) {
    let snapshot = &latest.0;
    let (state, detail, state_color) = state_words(snapshot);
    for (readout, mut text, mut color) in &mut readouts {
        match readout {
            Readout::Source => set(&mut text, source_words(snapshot)),
            Readout::State => {
                set(&mut text, state.to_owned());
                color.0 = state_color;
            }
            Readout::Detail => set(&mut text, detail.clone()),
            Readout::Breath => set(&mut text, breath_words(snapshot)),
            Readout::Where => set(&mut text, where_words(snapshot, &figure)),
            Readout::Drift => set(
                &mut text,
                match snapshot.moving_above {
                    Some(above) => format!("DASHED: MOVING ABOVE {above:.2}"),
                    None => "LEARNING THE QUIET LEVEL".into(),
                },
            ),
            Readout::Frames => {}
        }
    }
    for sparkline in &sparklines {
        if let Some(image) = images.get_mut(&sparkline.0) {
            raster::paint_sparkline(image, &snapshot.drift_history, DRIFT_SECONDS_SHOWN);
        }
    }
}

fn update_strip(
    latest: Res<Latest>,
    strip: Res<Strip>,
    mut images: ResMut<Assets<Image>>,
    mut rates: Query<(&LinkRate, &mut Text, &mut TextColor)>,
    mut bars: Query<(&LinkBar, &mut Node, &mut BackgroundColor)>,
) {
    let snapshot = &latest.0;
    for (index, handle) in strip.waterfalls.iter().enumerate() {
        let (Some(link), Some(image)) = (snapshot.links.get(index), images.get_mut(handle)) else { continue };
        raster::paint_waterfall(image, &link.csi_db, WATERFALL_PACKETS);
    }
    for (rate, mut text, mut color) in &mut rates {
        let Some(link) = snapshot.links.get(rate.0) else { continue };
        let strength = link.rssi.map_or(String::new(), |rssi| format!("  {rssi} DBM"));
        set(&mut text, format!("{:.0} PKT/S{strength}", link.packets_per_second));
        color.0 = if link.packets_per_second >= 10.0 { MUTED } else { EDGE };
    }
    let most_disturbed = snapshot.most_disturbed_link(MOST_DISTURBED_ABOVE);
    for (bar, mut node, mut color) in &mut bars {
        let activity = snapshot.links.get(bar.0).map_or(0.0, |l| l.activity);
        node.width = Val::Percent(100.0 * activity.clamp(0.0, 1.0));
        color.0 = if most_disturbed == Some(bar.0) { RED } else { MUTED };
    }
}

fn place_labels(
    latest: Res<Latest>,
    figure: Res<Figure>,
    cameras: Query<(&Camera, &GlobalTransform)>,
    mut labels: Query<(&WorldLabel, &mut Node, &mut Text, &mut Visibility)>,
) {
    let Ok((camera, eye)) = cameras.single() else { return };
    let snapshot = &latest.0;
    for (label, mut node, mut text, mut visibility) in &mut labels {
        let anchor = match label {
            WorldLabel::Board(index) => snapshot.boards.get(*index).map(|b| world(b.at)),
            WorldLabel::Person => {
                set(&mut text, person_words(snapshot, &figure));
                figure.at.filter(|_| figure.presence > 0.3).map(|_| figure.head + Vec3::Y * 0.25)
            }
        };
        match anchor.and_then(|at| camera.world_to_viewport(eye, at).ok()) {
            Some(at) => {
                node.left = Val::Px((at.x + 2.0 * PIXEL).round());
                node.top = Val::Px((at.y - PIXEL / 2.0).round());
                *visibility = Visibility::Inherited;
            }
            None => *visibility = Visibility::Hidden,
        }
    }
}

fn person_words(snapshot: &Snapshot, figure: &Figure) -> String {
    let doing = match figure.gait {
        Gait::Walk => "WALKING",
        Gait::Sit => "SITTING",
        Gait::Lie => "LYING",
        Gait::Stand => "STANDING",
    };
    match snapshot.breath_bpm {
        Some(bpm) => format!("{doing}  {bpm:.0}/MIN"),
        None => doing.to_owned(),
    }
}

fn count_frames(time: Res<Time>, mut shown: Local<f32>, mut readouts: Query<(&Readout, &mut Text)>) {
    let dt = time.delta_secs().max(1e-4);
    *shown += (1.0 / dt - *shown) * 0.05;
    for (readout, mut text) in &mut readouts {
        if matches!(readout, Readout::Frames) {
            set(&mut text, format!("{:.0} FPS", *shown));
        }
    }
}
