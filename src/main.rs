use gtk4::cairo;
use gtk4::glib;
use gtk4::prelude::*;
use gtk4::{
    Adjustment, Application, ApplicationWindow, Box as GtkBox, Button, DrawingArea, DropDown,
    Entry, FileChooserAction, FileChooserNative, FileFilter, Label, Orientation, ResponseType,
    SpinButton,
};
use midir::{Ignore, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
use rodio::{DeviceSinkBuilder, Player, buffer::SamplesBuffer};
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};
use std::cell::RefCell;
use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::num::{NonZeroU16, NonZeroU32};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Copy)]
struct Note {
    pitch: u8,
    onset_ms: u64,
    duration_ms: u64,
    velocity: u8,
    generated: bool,
}

#[derive(Default)]
struct Shared {
    notes: Vec<Note>,
    status: String,
}

#[derive(Default)]
struct PlaybackControl {
    generation: AtomicU64,
    playing: AtomicBool,
    player: Mutex<Option<Arc<Player>>>,
}

impl PlaybackControl {
    fn begin(&self) -> u64 {
        self.playing.store(true, Ordering::SeqCst);
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(player) = self.player.lock().unwrap().take() {
            player.stop();
        }
        generation
    }

    fn stop(&self) {
        self.playing.store(false, Ordering::SeqCst);
        self.generation.fetch_add(1, Ordering::SeqCst);
        if let Some(player) = self.player.lock().unwrap().take() {
            player.stop();
        }
    }

    fn is_active(&self, generation: u64) -> bool {
        self.generation.load(Ordering::SeqCst) == generation
    }

    fn is_playing(&self) -> bool {
        self.playing.load(Ordering::SeqCst)
    }

    fn set_player(&self, generation: u64, player: Arc<Player>) {
        let mut current = self.player.lock().unwrap();
        if self.is_active(generation) {
            *current = Some(player);
        } else {
            player.stop();
        }
    }

    fn finish(&self, generation: u64) {
        let mut current = self.player.lock().unwrap();
        if self.is_active(generation) {
            self.playing.store(false, Ordering::SeqCst);
            current.take();
        }
    }
}

#[derive(Clone, Copy)]
struct HeldNote {
    onset_ms: u64,
    velocity: u8,
    released: bool,
}

struct Capture {
    held: [Option<HeldNote>; 128],
    sustain: bool,
}

impl Capture {
    fn new() -> Self {
        Self {
            held: [None; 128],
            sustain: false,
        }
    }

    fn finish(&mut self, pitch: u8, end_ms: u64, shared: &Arc<Mutex<Shared>>) {
        if let Some(note) = self.held[pitch as usize].take() {
            shared.lock().unwrap().notes.push(Note {
                pitch,
                onset_ms: note.onset_ms,
                duration_ms: end_ms.saturating_sub(note.onset_ms).max(1),
                velocity: note.velocity,
                generated: false,
            });
        }
    }

    fn receive(&mut self, message: &[u8], now_ms: u64, shared: &Arc<Mutex<Shared>>) {
        if message.len() < 3 {
            return;
        }
        // ponytail: piano input treats all MIDI channels as one; split by channel if multi-instrument input matters.
        match message[0] & 0xf0 {
            0x90 if message[2] > 0 => {
                self.finish(message[1], now_ms, shared);
                self.held[message[1] as usize] = Some(HeldNote {
                    onset_ms: now_ms,
                    velocity: message[2],
                    released: false,
                });
            }
            0x80 | 0x90 => {
                let pitch = message[1];
                if self.sustain {
                    if let Some(note) = self.held[pitch as usize].as_mut() {
                        note.released = true;
                    }
                } else {
                    self.finish(pitch, now_ms, shared);
                }
            }
            0xb0 if message[1] == 64 => {
                let was_down = self.sustain;
                self.sustain = message[2] >= 64;
                if was_down && !self.sustain {
                    for pitch in 0..128 {
                        if self.held[pitch].is_some_and(|note| note.released) {
                            self.finish(pitch as u8, now_ms, shared);
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

struct GenerationRequest {
    checkpoint: PathBuf,
    prompt: String,
    bpm: f64,
}

struct ModelProcess {
    checkpoint: PathBuf,
    _child: Child,
    input: ChildStdin,
    output: BufReader<ChildStdout>,
}

impl ModelProcess {
    fn start(checkpoint: &Path) -> Result<Self, String> {
        let checkpoint = checkpoint
            .canonicalize()
            .map_err(|error| format!("Model not found: {error}"))?;
        let mut child = Command::new("uv")
            .args([
                "run",
                "--directory",
                "midilm",
                "--extra",
                "cpu",
                "midilm",
                "serve",
            ])
            .arg(&checkpoint)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map_err(|error| format!("Could not start midilm: {error}"))?;
        let input = child.stdin.take().ok_or("Could not open model input")?;
        let mut output = BufReader::new(child.stdout.take().ok_or("Could not open model output")?);
        let mut ready = String::new();
        output
            .read_line(&mut ready)
            .map_err(|error| error.to_string())?;
        if !ready.starts_with("ready\t") {
            return Err(format!("Model failed to start: {}", ready.trim()));
        }
        Ok(Self {
            checkpoint,
            _child: child,
            input,
            output,
        })
    }

    fn generate(&mut self, prompt: &str) -> Result<Vec<(u8, u64, u64, u8)>, String> {
        writeln!(self.input, "generate\t32\t0.9\t16\t{prompt}")
            .map_err(|error| error.to_string())?;
        self.input.flush().map_err(|error| error.to_string())?;
        let mut response = String::new();
        self.output
            .read_line(&mut response)
            .map_err(|error| error.to_string())?;
        let (kind, body) = response
            .trim_end()
            .split_once('\t')
            .unwrap_or(("error", "empty response"));
        if kind != "notes" {
            return Err(body.to_string());
        }
        if body.is_empty() {
            return Ok(Vec::new());
        }
        body.split(';')
            .map(|note| {
                let values = note
                    .split(',')
                    .map(str::parse::<u64>)
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(|error| error.to_string())?;
                if values.len() != 4 {
                    return Err("invalid model note".to_string());
                }
                Ok((values[0] as u8, values[1], values[2], values[3] as u8))
            })
            .collect()
    }
}

fn midi_inputs() -> Vec<String> {
    let Ok(input) = MidiInput::new("midi-autocomplete-list") else {
        return Vec::new();
    };
    input
        .ports()
        .iter()
        .filter_map(|port| input.port_name(port).ok())
        .collect()
}

fn midi_outputs() -> Vec<String> {
    let Ok(output) = MidiOutput::new("midi-autocomplete-list") else {
        return Vec::new();
    };
    output
        .ports()
        .iter()
        .filter_map(|port| output.port_name(port).ok())
        .collect()
}

fn dropdown(names: &[String]) -> DropDown {
    let labels: Vec<&str> = names.iter().map(String::as_str).collect();
    DropDown::from_strings(&labels)
}

fn connect_input(
    selected: u32,
    started: Instant,
    shared: Arc<Mutex<Shared>>,
) -> Result<MidiInputConnection<()>, String> {
    let mut input = MidiInput::new("midi-autocomplete-input").map_err(|error| error.to_string())?;
    input.ignore(Ignore::None);
    let ports = input.ports();
    let port = ports.get(selected as usize).ok_or("Select a MIDI input")?;
    let mut capture = Capture::new();
    input
        .connect(
            port,
            "midi-autocomplete-input",
            move |_, message, _| {
                capture.receive(message, started.elapsed().as_millis() as u64, &shared)
            },
            (),
        )
        .map_err(|error| error.to_string())
}

fn connect_output(selected: u32) -> Result<MidiOutputConnection, String> {
    let output = MidiOutput::new("midi-autocomplete-output").map_err(|error| error.to_string())?;
    let ports = output.ports();
    let port = ports.get(selected as usize).ok_or("Select a MIDI output")?;
    output
        .connect(port, "midi-autocomplete-output")
        .map_err(|error| error.to_string())
}

fn prompt(notes: &[Note], bpm: f64) -> String {
    let mut notes: Vec<_> = notes
        .iter()
        .filter(|note| !note.generated)
        .copied()
        .collect();
    notes.sort_by_key(|note| (note.onset_ms, note.pitch));
    let step_ms = 60_000.0 / bpm / 24.0;
    let mut previous = None;
    notes
        .into_iter()
        .map(|note| {
            let delta = previous.map_or(0, |onset| note.onset_ms.saturating_sub(onset));
            previous = Some(note.onset_ms);
            format!(
                "{},{},{},{}",
                note.pitch,
                (delta as f64 / step_ms).round() as u64,
                (note.duration_ms as f64 / step_ms).round().max(1.0) as u64,
                note.velocity
            )
        })
        .collect::<Vec<_>>()
        .join(";")
}

type MidiEvent = (u64, bool, u8, u8);

const SAMPLE_RATE: u32 = 48_000;
const MAX_PLAYBACK_MS: u64 = 5 * 60 * 1_000;

fn send_midi_events(
    events: Vec<MidiEvent>,
    started: Instant,
    output: Arc<Mutex<Option<MidiOutputConnection>>>,
    playback: Arc<PlaybackControl>,
    generation: u64,
) {
    for (time, on, pitch, velocity) in events {
        loop {
            if !playback.is_active(generation) {
                if let Some(connection) = output.lock().unwrap().as_mut() {
                    let _ = connection.send(&[0xb0, 120, 0]);
                    let _ = connection.send(&[0xb0, 123, 0]);
                }
                return;
            }
            let remaining = time.saturating_sub(started.elapsed().as_millis() as u64);
            if remaining == 0 {
                break;
            }
            thread::sleep(Duration::from_millis(remaining.min(10)));
        }
        if let Some(connection) = output.lock().unwrap().as_mut() {
            let _ = connection.send(&[if on { 0x90 } else { 0x80 }, pitch, velocity]);
        }
    }
}

fn replay_events(notes: &[Note]) -> Result<Vec<MidiEvent>, String> {
    let first = notes
        .iter()
        .map(|note| note.onset_ms)
        .min()
        .ok_or("Nothing to play")?;
    let end = notes
        .iter()
        .map(|note| note.onset_ms + note.duration_ms)
        .max()
        .unwrap();
    if end.saturating_sub(first) > MAX_PLAYBACK_MS {
        return Err("Playback is limited to five minutes. Clear older notes first.".into());
    }
    let mut events = Vec::with_capacity(notes.len() * 2);
    for note in notes {
        let onset = 100 + note.onset_ms.saturating_sub(first);
        events.push((onset, true, note.pitch, note.velocity));
        events.push((onset + note.duration_ms, false, note.pitch, 0));
    }
    events.sort_by_key(|event| (event.0, event.1));
    Ok(events)
}

fn render_soundfont(path: &Path, events: &[MidiEvent]) -> Result<Vec<f32>, String> {
    let mut file =
        File::open(path).map_err(|error| format!("Could not open SoundFont: {error}"))?;
    let sound_font = Arc::new(
        SoundFont::new(&mut file).map_err(|error| format!("Could not load SoundFont: {error}"))?,
    );
    let mut settings = SynthesizerSettings::new(SAMPLE_RATE as i32);
    settings.maximum_polyphony = 128;
    settings.enable_reverb_and_chorus = true;
    let mut synth = Synthesizer::new(&sound_font, &settings)
        .map_err(|error| format!("Could not start synthesizer: {error}"))?;
    let total_ms = events.last().map_or(0, |event| event.0) + 2_000;
    let sample_count = (total_ms * SAMPLE_RATE as u64 / 1_000) as usize;
    // ponytail: offline rendering is simple and bounded to five minutes; stream if longer sessions matter.
    let mut left = vec![0.0; sample_count];
    let mut right = vec![0.0; sample_count];
    let mut position = 0;
    for &(time, on, pitch, velocity) in events {
        let next = (time * SAMPLE_RATE as u64 / 1_000) as usize;
        if next > position {
            synth.render(&mut left[position..next], &mut right[position..next]);
            position = next;
        }
        if on {
            synth.note_on(0, pitch as i32, velocity as i32);
        } else {
            synth.note_off(0, pitch as i32);
        }
    }
    synth.render(&mut left[position..], &mut right[position..]);
    let mut samples = Vec::with_capacity(sample_count * 2);
    for (left, right) in left.into_iter().zip(right) {
        samples.extend([left, right]);
    }
    Ok(samples)
}

fn replay(
    notes: Vec<Note>,
    soundfont: PathBuf,
    started: Instant,
    shared: Arc<Mutex<Shared>>,
    output: Arc<Mutex<Option<MidiOutputConnection>>>,
    playback: Arc<PlaybackControl>,
) {
    let generation = playback.begin();
    shared.lock().unwrap().status = "Rendering SoundFont...".into();
    thread::spawn(move || {
        let events = match replay_events(&notes) {
            Ok(events) => events,
            Err(error) => {
                if playback.is_active(generation) {
                    shared.lock().unwrap().status = error;
                }
                playback.finish(generation);
                return;
            }
        };
        let samples = match render_soundfont(&soundfont, &events) {
            Ok(samples) => samples,
            Err(error) => {
                if playback.is_active(generation) {
                    shared.lock().unwrap().status = error;
                }
                playback.finish(generation);
                return;
            }
        };
        if !playback.is_active(generation) {
            return;
        }
        let device = match DeviceSinkBuilder::open_default_sink() {
            Ok(device) => device,
            Err(error) => {
                if playback.is_active(generation) {
                    shared.lock().unwrap().status = format!("Could not open audio output: {error}");
                }
                playback.finish(generation);
                return;
            }
        };
        let player = Arc::new(Player::connect_new(device.mixer()));
        playback.set_player(generation, player.clone());
        if !playback.is_active(generation) {
            return;
        }
        let base = started.elapsed().as_millis() as u64;
        let midi_events = events
            .into_iter()
            .map(|(time, on, pitch, velocity)| (base + time, on, pitch, velocity))
            .collect();
        let midi_output = output.clone();
        let midi_playback = playback.clone();
        let midi = thread::spawn(move || {
            send_midi_events(midi_events, started, midi_output, midi_playback, generation)
        });
        shared.lock().unwrap().status = format!("Playing {} notes", notes.len());
        player.append(SamplesBuffer::new(
            NonZeroU16::new(2).unwrap(),
            NonZeroU32::new(SAMPLE_RATE).unwrap(),
            samples,
        ));
        player.sleep_until_end();
        let _ = midi.join();
        if playback.is_active(generation) {
            shared.lock().unwrap().status = "Ready".into();
        }
        playback.finish(generation);
    });
}

fn play_generated(
    generated: Vec<(u8, u64, u64, u8)>,
    bpm: f64,
    started: Instant,
    shared: Arc<Mutex<Shared>>,
    output: Arc<Mutex<Option<MidiOutputConnection>>>,
    playback: Arc<PlaybackControl>,
) {
    let generation = playback.begin();
    let step_ms = 60_000.0 / bpm / 24.0;
    let live_base = started.elapsed().as_millis() as u64 + 100;
    let musical_base = shared
        .lock()
        .unwrap()
        .notes
        .iter()
        .filter(|note| !note.generated)
        .map(|note| note.onset_ms)
        .max()
        .unwrap_or(0);
    let mut onset = musical_base;
    let mut notes: Vec<Note> = Vec::new();
    for (pitch, delta, duration, velocity) in generated {
        onset += (delta as f64 * step_ms).round() as u64;
        let duration_ms = (duration as f64 * step_ms).round().max(1.0) as u64;
        if let Some(previous) = notes.iter_mut().rev().find(|note| note.pitch == pitch)
            && previous.onset_ms + previous.duration_ms > onset
        {
            previous.duration_ms = onset.saturating_sub(previous.onset_ms).max(1);
        }
        notes.push(Note {
            pitch,
            onset_ms: onset,
            duration_ms,
            velocity,
            generated: true,
        });
    }
    let mut events = Vec::with_capacity(notes.len() * 2);
    for note in &notes {
        let live_onset = live_base + note.onset_ms.saturating_sub(musical_base);
        events.push((live_onset, true, note.pitch, note.velocity));
        events.push((live_onset + note.duration_ms, false, note.pitch, 0));
    }
    events.sort_by_key(|event| (event.0, event.1));
    {
        let mut state = shared.lock().unwrap();
        state.notes.extend(notes);
        state.status = format!("Playing {} generated notes", events.len() / 2);
    }
    thread::spawn(move || {
        send_midi_events(events, started, output, playback.clone(), generation);
        if playback.is_active(generation) {
            shared.lock().unwrap().status = "Ready".to_string();
        }
        playback.finish(generation);
    });
}

fn draw_roll(cr: &cairo::Context, width: i32, height: i32, notes: &[Note]) {
    cr.set_source_rgb(0.07, 0.08, 0.10);
    let _ = cr.paint();
    let end = notes
        .iter()
        .map(|note| note.onset_ms + note.duration_ms)
        .max()
        .unwrap_or(15_000)
        .max(15_000);
    let start = end.saturating_sub(15_000);
    for pitch in 21..=108 {
        let y = height as f64 * (108 - pitch) as f64 / 88.0;
        cr.set_source_rgba(1.0, 1.0, 1.0, if pitch % 12 == 0 { 0.10 } else { 0.025 });
        cr.rectangle(0.0, y, width as f64, 1.0);
        let _ = cr.fill();
    }
    for note in notes {
        if note.onset_ms + note.duration_ms < start {
            continue;
        }
        let x = (note.onset_ms.saturating_sub(start) as f64 / 15_000.0) * width as f64;
        let w = (note.duration_ms as f64 / 15_000.0 * width as f64).max(2.0);
        let y = height as f64 * (108_i32 - note.pitch as i32) as f64 / 88.0;
        if note.generated {
            cr.set_source_rgb(0.25, 0.80, 0.48);
        } else {
            cr.set_source_rgb(0.32, 0.58, 0.95);
        }
        cr.rectangle(x, y, w, (height as f64 / 88.0).max(3.0));
        let _ = cr.fill();
    }
}

fn build_ui(app: &Application) {
    let started = Instant::now();
    let shared = Arc::new(Mutex::new(Shared {
        status: "Connect MIDI devices".into(),
        ..Default::default()
    }));
    let output: Arc<Mutex<Option<MidiOutputConnection>>> = Arc::new(Mutex::new(None));
    let playback_control = Arc::new(PlaybackControl::default());
    let input_connection: Rc<RefCell<Option<MidiInputConnection<()>>>> =
        Rc::new(RefCell::new(None));
    let input_names = midi_inputs();
    let output_names = midi_outputs();
    let has_midi_output = !output_names.is_empty();
    let input_dropdown = dropdown(&input_names);
    let output_dropdown = dropdown(&output_names);
    let connect = Button::with_label("Connect");
    let clear = Button::with_label("Clear");
    let generate = Button::with_label("Autocomplete");
    let play = Button::with_label("Play");
    let stop = Button::with_label("Stop");
    stop.set_visible(false);
    let browse = Button::with_label("Browse");
    let soundfont = Entry::builder()
        .placeholder_text("Select a .sf2 SoundFont")
        .hexpand(true)
        .build();
    let model = Entry::builder()
        .text("midilm/checkpoints/small.pt")
        .hexpand(true)
        .build();
    let bpm = SpinButton::new(
        Some(&Adjustment::new(120.0, 30.0, 300.0, 1.0, 10.0, 0.0)),
        1.0,
        0,
    );
    let status = Label::new(None);
    status.set_xalign(0.0);
    let roll = DrawingArea::builder()
        .content_height(420)
        .hexpand(true)
        .vexpand(true)
        .build();

    let state_for_draw = shared.clone();
    roll.set_draw_func(move |_, cr, width, height| {
        draw_roll(cr, width, height, &state_for_draw.lock().unwrap().notes)
    });

    let input_for_connect = input_connection.clone();
    let output_for_connect = output.clone();
    let state_for_connect = shared.clone();
    let input_select = input_dropdown.clone();
    let output_select = output_dropdown.clone();
    connect.connect_clicked(move |_| {
        match connect_input(input_select.selected(), started, state_for_connect.clone()) {
            Ok(connection) => *input_for_connect.borrow_mut() = Some(connection),
            Err(error) => {
                state_for_connect.lock().unwrap().status = error;
                return;
            }
        }
        if has_midi_output {
            match connect_output(output_select.selected()) {
                Ok(connection) => {
                    *output_for_connect.lock().unwrap() = Some(connection);
                    state_for_connect.lock().unwrap().status =
                        "MIDI connected. Play a prompt.".into();
                }
                Err(error) => state_for_connect.lock().unwrap().status = error,
            }
        } else {
            state_for_connect.lock().unwrap().status =
                "MIDI input connected. SoundFont playback only.".into();
        }
    });

    let state_for_clear = shared.clone();
    clear.connect_clicked(move |_| state_for_clear.lock().unwrap().notes.clear());

    let soundfont_for_browse = soundfont.clone();
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
        chooser.connect_response(move |chooser, response| {
            if response == ResponseType::Accept
                && let Some(path) = chooser.file().and_then(|file| file.path())
            {
                entry.set_text(&path.to_string_lossy());
            }
        });
        chooser.show();
    });

    let state_for_play = shared.clone();
    let output_for_play = output.clone();
    let playback_for_play = playback_control.clone();
    let soundfont_for_play = soundfont.clone();
    play.connect_clicked(move |_| {
        let path = PathBuf::from(soundfont_for_play.text().as_str());
        if path.as_os_str().is_empty() {
            state_for_play.lock().unwrap().status = "Choose a .sf2 SoundFont first".into();
            return;
        }
        let notes = state_for_play.lock().unwrap().notes.clone();
        if notes.is_empty() {
            state_for_play.lock().unwrap().status = "Nothing to play".into();
            return;
        }
        replay(
            notes,
            path,
            started,
            state_for_play.clone(),
            output_for_play.clone(),
            playback_for_play.clone(),
        );
    });

    let state_for_stop = shared.clone();
    let playback_for_stop = playback_control.clone();
    stop.connect_clicked(move |_| {
        playback_for_stop.stop();
        state_for_stop.lock().unwrap().status = "Stopped".into();
    });

    let (requests, receiver) = mpsc::channel::<GenerationRequest>();
    let worker_state = shared.clone();
    let worker_output = output.clone();
    let worker_playback = playback_control.clone();
    thread::spawn(move || {
        let mut process: Option<ModelProcess> = None;
        while let Ok(request) = receiver.recv() {
            let restart = process.as_ref().is_none_or(|current| {
                current.checkpoint != request.checkpoint.canonicalize().unwrap_or_default()
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
                Ok(notes) => play_generated(
                    notes,
                    request.bpm,
                    started,
                    worker_state.clone(),
                    worker_output.clone(),
                    worker_playback.clone(),
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
    generate.connect_clicked(move |_| {
        let state = state_for_generate.lock().unwrap();
        let input_notes = state.notes.iter().filter(|note| !note.generated).count();
        if input_notes == 0 {
            drop(state);
            state_for_generate.lock().unwrap().status =
                "Play at least one complete note first".into();
            return;
        }
        let request = GenerationRequest {
            checkpoint: PathBuf::from(model_for_generate.text().as_str()),
            prompt: prompt(&state.notes, bpm_for_generate.value()),
            bpm: bpm_for_generate.value(),
        };
        drop(state);
        let _ = requests.send(request);
    });

    let roll_for_timer = roll.clone();
    let state_for_timer = shared.clone();
    let status_for_timer = status.clone();
    let stop_for_timer = stop.clone();
    let playback_for_timer = playback_control.clone();
    glib::timeout_add_local(Duration::from_millis(33), move || {
        let state = state_for_timer.lock().unwrap();
        status_for_timer.set_text(&format!("{}  |  {} notes", state.status, state.notes.len()));
        drop(state);
        stop_for_timer.set_visible(playback_for_timer.is_playing());
        roll_for_timer.queue_draw();
        glib::ControlFlow::Continue
    });

    let devices = GtkBox::new(Orientation::Horizontal, 8);
    devices.append(&Label::new(Some("Input")));
    devices.append(&input_dropdown);
    devices.append(&Label::new(Some("Output")));
    devices.append(&output_dropdown);
    devices.append(&connect);

    let controls = GtkBox::new(Orientation::Horizontal, 8);
    controls.append(&Label::new(Some("Model")));
    controls.append(&model);
    controls.append(&Label::new(Some("BPM")));
    controls.append(&bpm);
    controls.append(&generate);
    controls.append(&clear);

    let playback = GtkBox::new(Orientation::Horizontal, 8);
    playback.append(&Label::new(Some("SoundFont")));
    playback.append(&soundfont);
    playback.append(&browse);
    playback.append(&play);
    playback.append(&stop);

    let content = GtkBox::new(Orientation::Vertical, 8);
    content.set_margin_top(12);
    content.set_margin_bottom(12);
    content.set_margin_start(12);
    content.set_margin_end(12);
    content.append(&devices);
    content.append(&controls);
    content.append(&playback);
    content.append(&roll);
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

fn main() -> glib::ExitCode {
    let app = Application::builder()
        .application_id("com.skorotkiewicz.midi-autocomplete")
        .build();
    app.connect_activate(build_ui);
    app.run()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replay_normalizes_the_musical_timeline() {
        let notes = [
            Note {
                pitch: 60,
                onset_ms: 500,
                duration_ms: 250,
                velocity: 80,
                generated: false,
            },
            Note {
                pitch: 64,
                onset_ms: 1_000,
                duration_ms: 100,
                velocity: 90,
                generated: true,
            },
        ];
        assert_eq!(
            replay_events(&notes).unwrap(),
            vec![
                (100, true, 60, 80),
                (350, false, 60, 0),
                (600, true, 64, 90),
                (700, false, 64, 0),
            ]
        );
    }

    #[test]
    fn stop_cancels_scheduled_midi_without_waiting() {
        let playback = Arc::new(PlaybackControl::default());
        let generation = playback.begin();
        assert!(playback.is_playing());
        playback.stop();
        assert!(!playback.is_playing());
        let started = Instant::now();
        send_midi_events(
            vec![(10_000, true, 60, 80)],
            started,
            Arc::new(Mutex::new(None)),
            playback,
            generation,
        );
        assert!(started.elapsed() < Duration::from_millis(100));
    }

    #[test]
    fn prompt_quantizes_at_24_steps_per_quarter() {
        let notes = [Note {
            pitch: 60,
            onset_ms: 500,
            duration_ms: 250,
            velocity: 80,
            generated: false,
        }];
        assert_eq!(prompt(&notes, 120.0), "60,0,12,80");
    }
}
