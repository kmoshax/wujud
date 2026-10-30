//! `HALLSENSE_SCREENSHOT=out.ppm`: run any window for a few seconds, save what it shows, and quit.
//! For README pictures and for checking the 3D view without a person at the screen.

use std::time::{Duration, Instant};

use eframe::egui;

const SETTLE: Duration = Duration::from_secs(25);

pub struct Screenshot<A> {
    app: A,
    path: String,
    started: Instant,
    requested: bool,
}

pub fn wrap<A: eframe::App + 'static>(app: A) -> Box<dyn eframe::App> {
    match std::env::var("HALLSENSE_SCREENSHOT") {
        Ok(path) => Box::new(Screenshot {
            app,
            path,
            started: Instant::now(),
            requested: false,
        }),
        Err(_) => Box::new(app),
    }
}

impl<A: eframe::App> eframe::App for Screenshot<A> {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        self.app.update(ctx, frame);
        if !self.requested && self.started.elapsed() > SETTLE {
            self.requested = true;
            ctx.send_viewport_cmd(egui::ViewportCommand::Screenshot(Default::default()));
        }
        let image = ctx.input(|i| {
            i.events.iter().find_map(|e| match e {
                egui::Event::Screenshot { image, .. } => Some(image.clone()),
                _ => None,
            })
        });
        if let Some(image) = image {
            if let Err(error) = write_ppm(&self.path, &image) {
                eprintln!("{}: {error}", self.path);
            }
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }
}

fn write_ppm(path: &str, image: &egui::ColorImage) -> std::io::Result<()> {
    let [width, height] = image.size;
    let mut bytes = format!("P6\n{width} {height}\n255\n").into_bytes();
    bytes.extend(image.pixels.iter().flat_map(|p| [p.r(), p.g(), p.b()]));
    std::fs::write(path, bytes)
}
