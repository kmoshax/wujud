//! `wujud view`: the live room in Bevy. A flat, dark 3D room with the boards, their links as beams,
//! the Fresnel zones that place the person, the skeleton, and the raw CSI the networks read.

mod body;
mod hud;
mod orbit;
mod palette;
mod person;
mod raster;
mod room;

use std::path::PathBuf;
use std::time::{Duration, Instant};

use bevy::prelude::*;
use bevy::render::view::screenshot::{Screenshot, ScreenshotCaptured};
use bevy::window::{MonitorSelection, PresentMode, WindowMode, WindowResolution};
use bevy::winit::{UpdateMode, WinitSettings};

use crate::live_core::{LiveCore, Snapshot};

/// Packets of CSI shown per link in the waterfall, newest on the right (~3 s at 100 packets/s).
pub const WATERFALL_PACKETS: usize = 300;

pub struct ViewOptions {
    pub max_fps: f64,
    pub fullscreen: bool,
    /// The camera view to start in (1-4, as the keys).
    pub first_view: usize,
    /// Render this many seconds, save the frame there and quit.
    pub screenshot: Option<(PathBuf, f32)>,
}

#[derive(Resource)]
pub struct Latest(pub Snapshot);

#[derive(Resource)]
struct FramePacer {
    interval: Duration,
    next: Instant,
}

#[derive(Resource)]
struct ScreenshotPlan {
    path: PathBuf,
    after: f32,
    taken: bool,
}

pub fn run(core: LiveCore, options: ViewOptions) -> anyhow::Result<()> {
    let first = core.snapshot(WATERFALL_PACKETS);
    let mut app = App::new();
    app.add_plugins(DefaultPlugins.set(WindowPlugin {
        primary_window: Some(Window {
            title: "wujud".into(),
            resolution: WindowResolution::new(1600.0, 940.0),
            // With vsync a covered window waits on the compositor and Hyprland calls it "not
            // responding"; frames are paced by FramePacer instead.
            present_mode: PresentMode::AutoNoVsync,
            mode: if options.fullscreen {
                WindowMode::BorderlessFullscreen(MonitorSelection::Current)
            } else {
                WindowMode::Windowed
            },
            ..default()
        }),
        ..default()
    }))
    .insert_resource(WinitSettings { focused_mode: UpdateMode::Continuous, unfocused_mode: UpdateMode::Continuous })
    .insert_resource(ClearColor(palette::BACKGROUND))
    .insert_resource(Latest(first))
    .insert_resource(orbit::FirstView(options.first_view))
    .insert_resource(FramePacer {
        interval: Duration::from_secs_f64(1.0 / options.max_fps.max(10.0)),
        next: Instant::now(),
    })
    .insert_non_send_resource(core)
    .add_plugins((orbit::OrbitPlugin, room::RoomPlugin, person::PersonPlugin, body::BodyPlugin, hud::HudPlugin))
    .add_systems(PreUpdate, pump_core)
    .add_systems(Last, pace_frames);
    if let Some((path, after)) = options.screenshot {
        app.insert_resource(ScreenshotPlan { path, after, taken: false })
            .add_systems(Update, take_screenshot);
    }
    app.run();
    Ok(())
}

fn pump_core(mut core: NonSendMut<LiveCore>, mut latest: ResMut<Latest>) {
    core.tick();
    latest.0 = core.snapshot(WATERFALL_PACKETS);
}

/// Sleeps out the rest of each frame's share, so the view runs at a steady rate without vsync.
fn pace_frames(mut pacer: ResMut<FramePacer>) {
    let now = Instant::now();
    if pacer.next > now {
        std::thread::sleep(pacer.next - now);
    }
    let interval = pacer.interval;
    pacer.next = (pacer.next + interval).max(Instant::now());
}

fn take_screenshot(mut commands: Commands, time: Res<Time>, mut plan: ResMut<ScreenshotPlan>) {
    if plan.taken || time.elapsed_secs() < plan.after {
        return;
    }
    plan.taken = true;
    let path = plan.path.clone();
    commands.spawn(Screenshot::primary_window()).observe(
        move |captured: Trigger<ScreenshotCaptured>, mut exit: EventWriter<AppExit>| {
            match captured.event().0.clone().try_into_dynamic() {
                Ok(image) => match image.to_rgb8().save(&path) {
                    Ok(()) => println!("saved {}", path.display()),
                    Err(error) => eprintln!("{}: {error}", path.display()),
                },
                Err(error) => eprintln!("screenshot: {error}"),
            }
            exit.write(AppExit::Success);
        },
    );
}

/// The simulator's metres (x across, y along, z up) in Bevy's world (y up).
pub fn world([x, y, z]: [f32; 3]) -> Vec3 {
    Vec3::new(x, z, -y)
}

pub fn floor([x, y]: [f32; 2], height: f32) -> Vec3 {
    world([x, y, height])
}

/// Eases towards a target at `rate` per second, independent of the frame rate.
pub fn ease(rate: f32, seconds: f32) -> f32 {
    1.0 - (-rate * seconds).exp()
}
