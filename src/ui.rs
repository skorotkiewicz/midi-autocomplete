use crate::config::{AppConfig, config_path, load_config, save_config};
use crate::midi::{connect_devices, midi_inputs, midi_outputs, silence_output};
use crate::model::{GenerationRequest, ModelProcess, prompt, resolve_model_path};
use crate::playback::{PlaybackControl, add_generated, replay};
use crate::state::Shared;
use crate::timeline::{PIXELS_PER_MS, draw_roll, timeline_bounds};
use gtk4::glib;
use gtk4::prelude::*;
use gtk4::{
    Adjustment, Application, ApplicationWindow, Box as GtkBox, Button, DrawingArea, DropDown,
    Entry, FileChooserAction, FileChooserNative, FileFilter, GestureClick, Label, Orientation,
    PolicyType, ResponseType, ScrolledWindow, SpinButton,
};
use midir::{MidiInputConnection, MidiOutputConnection};
use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

pub(crate) fn preferred_index(names: &[String], preferred: Option<&str>) -> Option<u32> {
    preferred.and_then(|name| {
        names
            .iter()
            .position(|candidate| candidate == name)
            .map(|index| index as u32)
    })
}

fn dropdown(names: &[String], preferred: Option<&str>) -> DropDown {
    let labels: Vec<&str> = names.iter().map(String::as_str).collect();
    let dropdown = DropDown::from_strings(&labels);
    if let Some(index) = preferred_index(names, preferred) {
        dropdown.set_selected(index);
    }
    dropdown
}

fn selected_name(dropdown: &DropDown) -> Option<String> {
    dropdown
        .selected_item()?
        .downcast::<gtk4::StringObject>()
        .ok()
        .map(|item| item.string().to_string())
}

fn refresh_dropdown(dropdown: &DropDown, names: &[String], preferred: Option<&str>) {
    let labels: Vec<&str> = names.iter().map(String::as_str).collect();
    dropdown.set_model(Some(&gtk4::StringList::new(&labels)));
    if let Some(index) = preferred_index(names, preferred) {
        dropdown.set_selected(index);
    }
}

pub(crate) fn build_ui(app: &Application) {
    let started = Instant::now();
    let config_path = config_path();
    let (initial_config, initial_status) = match load_config(&config_path) {
        Ok(config) => (config, "Connect MIDI devices".to_string()),
        Err(error) => (AppConfig::default(), error),
    };
    let shared = Arc::new(Mutex::new(Shared {
        status: initial_status,
        ..Default::default()
    }));
    let output: Arc<Mutex<Option<MidiOutputConnection>>> = Arc::new(Mutex::new(None));
    let playback_control = Arc::new(PlaybackControl::default());
    let input_connection: Rc<RefCell<Option<MidiInputConnection<()>>>> =
        Rc::new(RefCell::new(None));
    let input_names = midi_inputs();
    let output_names = midi_outputs();
    let input_dropdown = dropdown(&input_names, initial_config.midi_input.as_deref());
    let output_dropdown = dropdown(&output_names, initial_config.midi_output.as_deref());
    let connect = Button::with_label("Connect");
    let refresh = Button::with_label("Refresh");
    let record = Button::with_label("Rec");
    let clear = Button::with_label("Clear");
    let generate = Button::with_label("Autocomplete");
    let play = Button::with_label("Play");
    let pause = Button::with_label("Pause");
    let stop = Button::with_label("Stop");
    pause.set_visible(false);
    stop.set_visible(false);
    let browse = Button::with_label("Browse");
    let soundfont = Entry::builder()
        .placeholder_text("Select a .sf2 SoundFont")
        .hexpand(true)
        .build();
    if let Some(path) = &initial_config.soundfont {
        soundfont.set_text(path);
    }
    let model = Entry::builder()
        .text(
            initial_config
                .model
                .as_deref()
                .unwrap_or("midilm/checkpoints/medium.pt"),
        )
        .hexpand(true)
        .build();
    let config = Arc::new(Mutex::new(initial_config));
    let model_browse = Button::with_label("Browse");
    let bpm = SpinButton::new(
        Some(&Adjustment::new(120.0, 30.0, 300.0, 1.0, 10.0, 0.0)),
        1.0,
        0,
    );
    let status = Label::new(None);
    status.set_xalign(0.0);
    let roll = DrawingArea::builder()
        .content_height(420)
        .content_width(1_000)
        .hexpand(true)
        .vexpand(true)
        .build();
    let timeline = ScrolledWindow::builder()
        .hscrollbar_policy(PolicyType::Automatic)
        .vscrollbar_policy(PolicyType::Never)
        .vexpand(true)
        .child(&roll)
        .build();

    let state_for_draw = shared.clone();
    let playback_for_draw = playback_control.clone();
    roll.set_draw_func(move |_, cr, width, height| {
        let now = started.elapsed().as_millis() as u64;
        let state = state_for_draw.lock().unwrap();
        let playhead = if playback_for_draw.is_playing() {
            playback_for_draw.position(now)
        } else if state.capturing {
            Some(state.capture_position(now))
        } else {
            playback_for_draw.cursor()
        };
        draw_roll(cr, width, height, &state.notes, playhead);
    });

    let config_for_input = config.clone();
    let path_for_input = config_path.clone();
    let state_for_input = shared.clone();
    input_dropdown.connect_selected_notify(move |dropdown| {
        let snapshot = {
            let mut config = config_for_input.lock().unwrap();
            config.midi_input = selected_name(dropdown);
            config.clone()
        };
        if let Err(error) = save_config(&path_for_input, &snapshot) {
            state_for_input.lock().unwrap().status = error;
        }
    });

    let config_for_output = config.clone();
    let path_for_output = config_path.clone();
    let state_for_output = shared.clone();
    output_dropdown.connect_selected_notify(move |dropdown| {
        let snapshot = {
            let mut config = config_for_output.lock().unwrap();
            config.midi_output = selected_name(dropdown);
            config.clone()
        };
        if let Err(error) = save_config(&path_for_output, &snapshot) {
            state_for_output.lock().unwrap().status = error;
        }
    });

    let input_for_connect = input_connection.clone();
    let output_for_connect = output.clone();
    let state_for_connect = shared.clone();
    let config_for_connect = config.clone();
    let path_for_connect = config_path.clone();
    let input_select = input_dropdown.clone();
    let output_select = output_dropdown.clone();
    connect.connect_clicked(move |_| {
        match connect_devices(
            input_select.selected(),
            output_select.selected(),
            started,
            state_for_connect.clone(),
            &input_for_connect,
            &output_for_connect,
        ) {
            Ok(()) => {
                let snapshot = {
                    let mut config = config_for_connect.lock().unwrap();
                    config.midi_input = selected_name(&input_select);
                    config.midi_output = selected_name(&output_select);
                    config.clone()
                };
                let mut state = state_for_connect.lock().unwrap();
                state.connected = true;
                state.capturing = false;
                state.capture_generation += 1;
                state.status = match save_config(&path_for_connect, &snapshot) {
                    Ok(()) => "MIDI connected. Click Rec to capture a prompt.".into(),
                    Err(error) => error,
                };
            }
            Err(error) => state_for_connect.lock().unwrap().status = error,
        }
    });

    let input_for_refresh = input_connection.clone();
    let output_for_refresh = output.clone();
    let state_for_refresh = shared.clone();
    let input_select = input_dropdown.clone();
    let output_select = output_dropdown.clone();
    refresh.connect_clicked(move |_| {
        let preferred_input = selected_name(&input_select);
        let preferred_output = selected_name(&output_select);
        silence_output(&output_for_refresh);
        *input_for_refresh.borrow_mut() = None;
        *output_for_refresh.lock().unwrap() = None;
        let inputs = midi_inputs();
        let outputs = midi_outputs();
        refresh_dropdown(&input_select, &inputs, preferred_input.as_deref());
        refresh_dropdown(&output_select, &outputs, preferred_output.as_deref());
        let mut state = state_for_refresh.lock().unwrap();
        state.connected = false;
        state.capturing = false;
        state.capture_generation += 1;
        state.status = format!(
            "Found {} inputs and {} outputs. Click Connect.",
            inputs.len(),
            outputs.len()
        );
    });

    let auto_connect = {
        let config = config.lock().unwrap();
        config.midi_input.as_ref() == selected_name(&input_dropdown).as_ref()
            && config.midi_output.as_ref() == selected_name(&output_dropdown).as_ref()
            && config.midi_input.is_some()
            && config.midi_output.is_some()
    };
    if auto_connect {
        match connect_devices(
            input_dropdown.selected(),
            output_dropdown.selected(),
            started,
            shared.clone(),
            &input_connection,
            &output,
        ) {
            Ok(()) => {
                let mut state = shared.lock().unwrap();
                state.connected = true;
                state.capturing = false;
                state.capture_generation += 1;
                state.status = "MIDI auto-connected. Click Rec to capture a prompt.".into();
            }
            Err(error) => shared.lock().unwrap().status = format!("Auto-connect failed: {error}"),
        }
    }

    let state_for_record = shared.clone();
    let playback_for_record = playback_control.clone();
    record.connect_clicked(move |_| {
        let selected_position = {
            let mut state = state_for_record.lock().unwrap();
            if !state.connected {
                state.status = "Connect MIDI devices before recording.".into();
                return;
            }
            let selected_position = (!state.capturing)
                .then(|| playback_for_record.cursor())
                .flatten();
            state.toggle_capture(started.elapsed().as_millis() as u64, selected_position);
            selected_position
        };
        if selected_position.is_some() {
            playback_for_record.reset();
        }
    });

    let state_for_clear = shared.clone();
    let playback_for_clear = playback_control.clone();
    clear.connect_clicked(move |_| {
        playback_for_clear.reset();
        state_for_clear
            .lock()
            .unwrap()
            .clear_timeline(started.elapsed().as_millis() as u64)
    });

    let soundfont_for_browse = soundfont.clone();
    let config_for_browse = config.clone();
    let path_for_browse = config_path.clone();
    let state_for_browse = shared.clone();
    browse.connect_clicked(move |_| {
        let chooser = FileChooserNative::builder()
            .title("Choose a SoundFont")
            .action(FileChooserAction::Open)
            .accept_label("Open")
            .build();
        let filter = FileFilter::new();
        filter.set_name(Some("SoundFont files"));
        filter.add_pattern("*.sf2");
        filter.add_pattern("*.SF2");
        chooser.set_filter(&filter);
        let entry = soundfont_for_browse.clone();
        let config = config_for_browse.clone();
        let config_path = path_for_browse.clone();
        let state = state_for_browse.clone();
        chooser.connect_response(move |chooser, response| {
            if response == ResponseType::Accept
                && let Some(path) = chooser.file().and_then(|file| file.path())
            {
                let path = path.to_string_lossy().into_owned();
                entry.set_text(&path);
                let snapshot = {
                    let mut config = config.lock().unwrap();
                    config.soundfont = Some(path);
                    config.clone()
                };
                if let Err(error) = save_config(&config_path, &snapshot) {
                    state.lock().unwrap().status = error;
                }
            }
        });
        chooser.show();
    });

    let model_for_browse = model.clone();
    let config_for_model_browse = config.clone();
    let path_for_model_browse = config_path.clone();
    let state_for_model_browse = shared.clone();
    model_browse.connect_clicked(move |_| {
        let chooser = FileChooserNative::builder()
            .title("Choose a Model Checkpoint")
            .action(FileChooserAction::Open)
            .accept_label("Open")
            .build();
        let filter = FileFilter::new();
        filter.set_name(Some("PyTorch checkpoints"));
        filter.add_pattern("*.pt");
        filter.add_pattern("*.pth");
        chooser.set_filter(&filter);
        let entry = model_for_browse.clone();
        let config = config_for_model_browse.clone();
        let config_path = path_for_model_browse.clone();
        let state = state_for_model_browse.clone();
        chooser.connect_response(move |chooser, response| {
            if response == ResponseType::Accept
                && let Some(path) = chooser.file().and_then(|file| file.path())
            {
                let path = path.to_string_lossy().into_owned();
                entry.set_text(&path);
                let snapshot = {
                    let mut config = config.lock().unwrap();
                    config.model = Some(path);
                    config.clone()
                };
                if let Err(error) = save_config(&config_path, &snapshot) {
                    state.lock().unwrap().status = error;
                }
            }
        });
        chooser.show();
    });

    let state_for_play = shared.clone();
    let output_for_play = output.clone();
    let playback_for_play = playback_control.clone();
    let soundfont_for_play = soundfont.clone();
    let config_for_play = config.clone();
    let path_for_play = config_path.clone();
    play.connect_clicked(move |_| {
        if playback_for_play.is_rendering() {
            state_for_play.lock().unwrap().status = "SoundFont is still rendering...".into();
            return;
        }
        let path = PathBuf::from(soundfont_for_play.text().as_str());
        if path.as_os_str().is_empty() {
            state_for_play.lock().unwrap().status = "Choose a .sf2 SoundFont first".into();
            return;
        }
        let snapshot = {
            let mut config = config_for_play.lock().unwrap();
            config.soundfont = Some(path.to_string_lossy().into_owned());
            config.clone()
        };
        if let Err(error) = save_config(&path_for_play, &snapshot) {
            state_for_play.lock().unwrap().status = error;
            return;
        }
        let notes = state_for_play.lock().unwrap().notes.clone();
        if notes.is_empty() {
            state_for_play.lock().unwrap().status = "Nothing to play".into();
            return;
        }
        let musical_start_ms = playback_for_play.cursor();
        replay(
            notes,
            path,
            musical_start_ms,
            started,
            state_for_play.clone(),
            output_for_play.clone(),
            playback_for_play.clone(),
        );
    });

    let state_for_pause = shared.clone();
    let playback_for_pause = playback_control.clone();
    pause.connect_clicked(move |_| {
        if playback_for_pause.pause(started.elapsed().as_millis() as u64) {
            state_for_pause.lock().unwrap().status = "Paused".into();
        }
    });

    let state_for_stop = shared.clone();
    let playback_for_stop = playback_control.clone();
    stop.connect_clicked(move |_| {
        playback_for_stop.stop(started.elapsed().as_millis() as u64);
        state_for_stop.lock().unwrap().status = "Stopped".into();
    });

    let seek = GestureClick::new();
    let state_for_seek = shared.clone();
    let output_for_seek = output.clone();
    let playback_for_seek = playback_control.clone();
    let soundfont_for_seek = soundfont.clone();
    seek.connect_pressed(move |_, _, x, _| {
        let now = started.elapsed().as_millis() as u64;
        let (notes, bounds) = {
            let state = state_for_seek.lock().unwrap();
            let playhead = playback_for_seek
                .position(now)
                .or_else(|| playback_for_seek.cursor())
                .or_else(|| state.capturing.then(|| state.capture_position(now)));
            (state.notes.clone(), timeline_bounds(&state.notes, playhead))
        };
        let Some((start, end)) = bounds else {
            return;
        };
        let position = (start + (x.max(0.0) / PIXELS_PER_MS) as u64).min(end);
        let was_playing = playback_for_seek.seek(position);
        state_for_seek.lock().unwrap().status =
            format!("Position: {:.2}s", position as f64 / 1_000.0);
        if !was_playing {
            return;
        }
        let path = PathBuf::from(soundfont_for_seek.text().as_str());
        if path.as_os_str().is_empty() {
            state_for_seek.lock().unwrap().status = "Choose a .sf2 SoundFont first".into();
            return;
        }
        replay(
            notes,
            path,
            Some(position),
            started,
            state_for_seek.clone(),
            output_for_seek.clone(),
            playback_for_seek.clone(),
        );
    });
    roll.add_controller(seek);

    let (requests, receiver) = mpsc::channel::<GenerationRequest>();
    let worker_state = shared.clone();
    thread::spawn(move || {
        let mut process: Option<ModelProcess> = None;
        while let Ok(request) = receiver.recv() {
            let restart = process.as_ref().is_none_or(|current| {
                current.checkpoint != resolve_model_path(&request.checkpoint)
            });
            if restart {
                process = match ModelProcess::start(&request.checkpoint) {
                    Ok(model) => Some(model),
                    Err(error) => {
                        worker_state.lock().unwrap().status = error;
                        continue;
                    }
                };
            }
            worker_state.lock().unwrap().status = "Generating...".into();
            match process.as_mut().unwrap().generate(&request.prompt) {
                Ok(notes) => add_generated(
                    notes,
                    request.bpm,
                    request.musical_start_ms,
                    worker_state.clone(),
                ),
                Err(error) => {
                    worker_state.lock().unwrap().status = error;
                    process = None;
                }
            }
        }
    });

    let state_for_generate = shared.clone();
    let model_for_generate = model.clone();
    let bpm_for_generate = bpm.clone();
    let config_for_generate = config.clone();
    let path_for_generate = config_path.clone();
    let playback_for_generate = playback_control.clone();
    generate.connect_clicked(move |_| {
        let musical_start_ms = playback_for_generate
            .position(started.elapsed().as_millis() as u64)
            .or_else(|| playback_for_generate.cursor());
        let state = state_for_generate.lock().unwrap();
        let prompt_text = prompt(&state.notes, bpm_for_generate.value(), musical_start_ms);
        if prompt_text.is_empty() {
            drop(state);
            state_for_generate.lock().unwrap().status =
                "Play at least one complete note before this position".into();
            return;
        }
        let model_path = model_for_generate.text().to_string();
        let snapshot = {
            let mut config = config_for_generate.lock().unwrap();
            config.model = Some(model_path.clone());
            config.clone()
        };
        if let Err(error) = save_config(&path_for_generate, &snapshot) {
            drop(state);
            state_for_generate.lock().unwrap().status = error;
            return;
        }
        let request = GenerationRequest {
            checkpoint: PathBuf::from(model_path),
            prompt: prompt_text,
            bpm: bpm_for_generate.value(),
            musical_start_ms,
        };
        drop(state);
        let _ = requests.send(request);
    });

    let roll_for_timer = roll.clone();
    let timeline_for_timer = timeline.clone();
    let state_for_timer = shared.clone();
    let status_for_timer = status.clone();
    let play_for_timer = play.clone();
    let pause_for_timer = pause.clone();
    let stop_for_timer = stop.clone();
    let record_for_timer = record.clone();
    let playback_for_timer = playback_control.clone();
    glib::timeout_add_local(Duration::from_millis(33), move || {
        let now = started.elapsed().as_millis() as u64;
        let state = state_for_timer.lock().unwrap();
        let playhead = if playback_for_timer.is_playing() {
            playback_for_timer.position(now)
        } else if state.capturing {
            Some(state.capture_position(now))
        } else {
            playback_for_timer.cursor()
        };
        let timeline_position =
            playhead.or_else(|| state.connected.then(|| state.capture_position(now)));
        let bounds = timeline_bounds(&state.notes, timeline_position);
        status_for_timer.set_text(&format!("{}  |  {} notes", state.status, state.notes.len()));
        drop(state);

        let viewport = timeline_for_timer.width().max(1);
        let content_width = bounds.map_or(viewport, |(start, end)| {
            ((end.saturating_sub(start) + 1_000) as f64 * PIXELS_PER_MS) as i32
        });
        roll_for_timer.set_content_width(content_width.max(viewport));
        if let (Some(playhead), Some((start, _))) = (playhead, bounds) {
            let x = playhead.saturating_sub(start) as f64 * PIXELS_PER_MS;
            let adjustment = timeline_for_timer.hadjustment();
            let page = viewport as f64;
            if x > adjustment.value() + page - 40.0 {
                adjustment.set_value((x - page + 40.0).max(0.0));
            } else if x < adjustment.value() {
                adjustment.set_value(x.max(0.0));
            }
        }
        record_for_timer.set_label(if state_for_timer.lock().unwrap().capturing {
            "Stop Rec"
        } else {
            "Rec"
        });
        let playing = playback_for_timer.is_playing();
        let paused = playback_for_timer.is_paused();
        play_for_timer.set_label(if paused { "Resume" } else { "Play" });
        play_for_timer.set_visible(!playing);
        play_for_timer.set_sensitive(!playback_for_timer.is_rendering());
        pause_for_timer.set_visible(playing);
        stop_for_timer.set_visible(playing || paused);
        roll_for_timer.queue_draw();
        glib::ControlFlow::Continue
    });

    let devices = GtkBox::new(Orientation::Horizontal, 8);
    devices.append(&Label::new(Some("Input")));
    devices.append(&input_dropdown);
    devices.append(&Label::new(Some("Output")));
    devices.append(&output_dropdown);
    devices.append(&connect);
    devices.append(&refresh);

    let controls = GtkBox::new(Orientation::Horizontal, 8);
    controls.append(&Label::new(Some("Model")));
    controls.append(&model);
    controls.append(&model_browse);
    controls.append(&Label::new(Some("BPM")));
    controls.append(&bpm);
    controls.append(&generate);
    controls.append(&clear);

    let playback = GtkBox::new(Orientation::Horizontal, 8);
    playback.append(&Label::new(Some("SoundFont")));
    playback.append(&soundfont);
    playback.append(&browse);
    playback.append(&record);
    playback.append(&play);
    playback.append(&pause);
    playback.append(&stop);

    let content = GtkBox::new(Orientation::Vertical, 8);
    content.set_margin_top(12);
    content.set_margin_bottom(12);
    content.set_margin_start(12);
    content.set_margin_end(12);
    content.append(&devices);
    content.append(&controls);
    content.append(&playback);
    content.append(&timeline);
    content.append(&status);

    ApplicationWindow::builder()
        .application(app)
        .title("MIDI Autocomplete")
        .default_width(1000)
        .default_height(600)
        .child(&content)
        .build()
        .present();
}
