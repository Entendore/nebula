use eframe::Result;
use std::sync::Arc;
use tokio::runtime::Runtime;

mod ai;
mod app;
mod highlight;

fn main() -> Result<(), eframe::Error> {
    let runtime = Arc::new(Runtime::new().expect("Failed to create Tokio runtime"));
    let config = app::AppConfig::load();

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([config.window_width, config.window_height])
            .with_min_inner_size([800.0, 600.0])
            .with_title("Nebula — AI Code Editor"),
        ..Default::default()
    };

    eframe::run_native(
        "Nebula",
        options,
        Box::new(move |cc| Box::new(app::NebulaApp::new(cc, runtime.clone()))),
    )
}
