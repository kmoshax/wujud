#[cfg(target_os = "espidf")]
mod board;

#[cfg(target_os = "espidf")]
fn main() -> anyhow::Result<()> {
    board::run()
}

#[cfg(not(target_os = "espidf"))]
fn main() {
    eprintln!("hallsense-firmware runs on an ESP32; see firmware/README.md for the build command.");
}
