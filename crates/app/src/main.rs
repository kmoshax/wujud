mod camera;
mod hall_feed;
mod hall_view;
mod hall_window;
mod link_history;
mod live_window;
mod room_setup;
mod room_view;
mod scene_3d;
mod screenshot;
mod sensor_feed;
mod serial;
mod udp;

use clap::{Parser, Subcommand};
use wujud_proto::{Line, LineError};

use live_window::LiveWindow;

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

fn main() -> anyhow::Result<()> {
    match Cli::parse().command {
        Command::Live {
            ports,
            room,
            record,
        } => live(ports, room.as_deref(), record.as_deref()),
        Command::Demo { boards, seed } => demo(boards, seed),
        Command::DemoHall { seed } => demo_hall(seed),
        Command::Monitor { ports, record } => monitor(&ports, record.as_deref()),
    }
}

fn live(
    ports: Vec<String>,
    room: Option<&std::path::Path>,
    record: Option<&str>,
) -> anyhow::Result<()> {
    let ports = serial::one_link_per_device(&ports);
    println!("links: {}", ports.join(", "));
    let room = room_setup::room_engine(&ports, room)?;
    let feed = sensor_feed::from_serial_ports(&ports, record)?;
    open_window(LiveWindow::new(&ports, feed, room))
}

fn demo(boards: usize, seed: u64) -> anyhow::Result<()> {
    let stream = wujud_sim::stream::RoomStream::demo(seed, boards);
    let room = room_setup::simulated_room_engine(stream.room())?;
    let names: Vec<String> = (1..=boards)
        .map(|k| format!("simulated ESP32 {k}"))
        .collect();
    open_window(LiveWindow::new(
        &names,
        sensor_feed::from_simulator(stream),
        room,
    ))
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
        ..Default::default()
    };
    eframe::run_native(
        "wujud",
        options,
        Box::new(move |_| Ok(screenshot::wrap(window))),
    )
    .map_err(|e| anyhow::anyhow!("{e}"))
}

fn monitor(ports: &[String], record: Option<&str>) -> anyhow::Result<()> {
    let ports = serial::one_link_per_device(ports);
    for line in serial::read_ports(&ports, record)? {
        let port = &ports[line.port];
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
