mod config;
mod midi;
mod midi_file;
mod model;
mod playback;
mod state;
mod timeline;
mod ui;

use gtk4::Application;
use gtk4::glib;
use gtk4::prelude::*;
use ui::build_ui;

fn main() -> glib::ExitCode {
    let app = Application::builder()
        .application_id("com.skorotkiewicz.midi-autocomplete")
        .build();
    app.connect_activate(build_ui);
    app.run()
}

#[cfg(test)]
mod tests;
