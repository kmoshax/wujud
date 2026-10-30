mod csi;
mod serial;

#[cfg(feature = "hall-node")]
mod hall_node;
#[cfg(all(feature = "room", not(feature = "hall-node")))]
mod room;

pub fn run() -> anyhow::Result<()> {
    esp_idf_svc::sys::link_patches();
    #[cfg(feature = "hall-node")]
    return hall_node::run();
    #[cfg(all(feature = "room", not(feature = "hall-node")))]
    return room::run();
}
