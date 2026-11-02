mod camera;
mod hall_feed;
mod hall_view;
mod hall_window;
mod learned_tracker;
mod link_net;
mod link_history;
mod live_core;
mod live_window;
mod phone_range;
mod room_sense;
mod room_setup;
mod room_view;
mod scene_3d;
mod screenshot;
mod sensor_feed;
mod serial;
mod skeleton;
mod theme;
mod udp;
mod view;

use clap::{Parser, Subcommand};
use wujud_proto::{Line, LineError};

use live_window::LiveWindow;

/// About 30 frames a second: smooth enough for the 3D view, light on the laptop.
pub const FRAME_INTERVAL: std::time::Duration = std::time::Duration::from_millis(33);

#[derive(Parser)]
#[command(name = "wujud", about = "Wi-Fi sensing with ESP32 boards")]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Open the live window for one or more boards.
    Live {
        /// Serial ports (/dev/ttyACM0) or boards on Wi-Fi (udp:192.168.4.1); each device a board
        /// measures becomes its own link (`udp:192.168.4.1@<mac>`).
        #[arg(long, num_args = 1.., required = true)]
        ports: Vec<String>,
        /// Room layout (research/room.example.json format); without it the boards go along the walls.
        #[arg(long)]
        room: Option<std::path::PathBuf>,
        #[arg(long)]
        record: Option<String>,
    },
    /// No boards needed: the simulator plays a room (empty, someone walks in, sits, lies down, leaves).
    Demo {
        #[arg(long, default_value_t = 1)]
        boards: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
    },
    /// The live room in 3D (Bevy): the boards, their links, the person, and the raw CSI.
    View {
        /// As for `live`.
        #[arg(long, num_args = 1.., required = true)]
        ports: Vec<String>,
        #[arg(long)]
        room: Option<std::path::PathBuf>,
        #[arg(long)]
        record: Option<String>,
        #[command(flatten)]
        view: ViewArgs,
    },
    /// The 3D view (Bevy) on the simulator: no boards needed.
    ViewDemo {
        #[arg(long, default_value_t = 3)]
        boards: usize,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        #[command(flatten)]
        view: ViewArgs,
    },
    /// No boards needed: the simulator plays a lecture hall filling up, counted by 12 simulated wall nodes.
    DemoHall {
        #[arg(long, default_value_t = 3)]
        seed: u64,
    },
    /// Print what the boards send, decoded, to check the wiring.
    Monitor {
        #[arg(long, num_args = 1.., required = true)]
        ports: Vec<String>,
        #[arg(long)]
        record: Option<String>,
    },
}

#[derive(clap::Args)]
struct ViewArgs {
    /// Frames a second at most (the view runs without vsync).
    #[arg(long, default_value_t = 120.0)]
    max_fps: f64,
    /// Start full screen (f toggles it).
    #[arg(long)]
    fullscreen: bool,
    /// The camera view to start in: 1 room, 2 from above, 3 low, 4 following the person.
    #[arg(long, default_value_t = 1)]
    camera: usize,
    /// Save one frame to this PNG after `--after` seconds, then quit.
    #[arg(long)]
    screenshot: Option<std::path::PathBuf>,
    #[arg(long, default_value_t = 25.0)]
    after: f32,
}

impl ViewArgs {
    fn options(self) -> view::ViewOptions {
        view::ViewOptions {
            max_fps: self.max_fps,
            fullscreen: self.fullscreen,
            first_view: self.camera,
            screenshot: self.screenshot.map(|path| (path, self.after)),
        }
    }
}

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Live {
            ports,
            room,
            record,
        } => live(ports, room.as_deref(), record.as_deref()),
        Command::Demo { boards, seed } => demo(boards, seed),
        Command::View {
            ports,
            room,
            record,
            view,
        } => live_view(ports, room.as_deref(), record.as_deref(), view.options()),
        Command::ViewDemo { boards, seed, view } => demo_view(boards, seed, view.options()),
        Command::DemoHall { seed } => demo_hall(seed),
        Command::Monitor { ports, record } => monitor(&ports, record.as_deref()),
    }
}

fn live(
    ports: Vec<String>,
    room: Option<&std::path::Path>,
    record: Option<&str>,
) -> anyhow::Result<()> {
    open_window(LiveWindow::new(boards_core(ports, room, record)?))
}

fn live_view(
    ports: Vec<String>,
    room: Option<&std::path::Path>,
    record: Option<&str>,
    options: view::ViewOptions,
) -> anyhow::Result<()> {
    view::run(boards_core(ports, room, record)?, options)
}

fn boards_core(
    ports: Vec<String>,
    room: Option<&std::path::Path>,
    record: Option<&str>,
) -> anyhow::Result<live_core::LiveCore> {
    let recorded = ports.iter().any(|port| port.starts_with("replay:"));
    let (ports, rssi) = serial::discover(&ports);
    let live = room_setup::room_engine(&ports, room, &rssi)?;
    println!("links: {}", live.links.join(", "));
    let reading = std::sync::Arc::new(std::sync::Mutex::new(live.links.clone()));
    let feed = sensor_feed::from_serial_ports(&ports, reading.clone(), record)?;
    Ok(live_core::LiveCore::new(feed, live).growing(live_core::Growth {
        seen: ports,
        reading,
        room_file: room.map(std::path::Path::to_path_buf),
        rssi,
        recorded,
    }))
}

fn simulated_core(boards: usize, seed: u64) -> anyhow::Result<live_core::LiveCore> {
    let stream = wujud_sim::stream::RoomStream::demo(seed, boards);
    let room = room_setup::simulated_room_engine(stream.room())?;
    Ok(live_core::LiveCore::new(sensor_feed::from_simulator(stream), room))
}

fn demo_view(boards: usize, seed: u64, options: view::ViewOptions) -> anyhow::Result<()> {
    view::run(simulated_core(boards, seed)?, options)
}

fn demo(boards: usize, seed: u64) -> anyhow::Result<()> {
    open_window(LiveWindow::new(simulated_core(boards, seed)?))
}

fn demo_hall(seed: u64) -> anyhow::Result<()> {
    let hall = hall_feed::simulate(wujud_sim::stream::HallStream::demo(seed))?;
    let model = wujud_model::hall::SeatModel::load(&room_setup::models_dir().join("hall3d.json"))?;
    let engine = wujud_model::hall::HallEngine::new(
        hall.layout,
        model,
        hall.calibration_mean,
        hall.calibration_std,
        "SIMULATION",
    );
    open_window(hall_window::HallWindow::new(hall.seconds, engine))
}

fn open_window(window: impl eframe::App + 'static) -> anyhow::Result<()> {
    let options = eframe::NativeOptions {
        viewport: eframe::egui::ViewportBuilder::default()
            .with_title("wujud")
            .with_inner_size([1200.0, 820.0]),
        // With vsync a hidden or covered window waits for the compositor without end, and Hyprland
        // calls it "not responding"; frames are paced by the window instead (FRAME_INTERVAL).
        vsync: false,
        ..Default::default()
    };
    eframe::run_native(
        "wujud",
        options,
        Box::new(move |creation| {
            theme::apply(&creation.egui_ctx);
            Ok(screenshot::wrap(window))
        }),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

fn monitor(ports: &[String], record: Option<&str>) -> anyhow::Result<()> {
    let ports = serial::one_link_per_device(ports);
    for line in serial::read_ports(&ports, record)? {
        let port = &line.link;
        match Line::parse(&line.text) {
            Ok(Line::Csi(packet)) => println!(
                "[{port}] CSI  rssi {:>4} dBm  t {:>10} us",
                packet.rssi, packet.rx_timestamp_us
            ),
            Ok(other) => println!("[{port}] {other}"),
            Err(LineError::Empty) => {}
            Err(error) => println!("[{port}] ? {error}: {}", line.text.trim_end()),
        }
    }
    Ok(())
}
