use gtk4::cairo;
use gtk4::glib;
use gtk4::prelude::*;
use gtk4::{
    Adjustment, Application, ApplicationWindow, Box as GtkBox, Button, DrawingArea, DropDown,
    Entry, FileChooserAction, FileChooserNative, FileFilter, Label, Orientation, PolicyType,
    ResponseType, ScrolledWindow, SpinButton,
};
use midir::{Ignore, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
use rodio::{DeviceSinkBuilder, Player, buffer::SamplesBuffer};
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs::{self, File};
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
    connected: bool,
    capturing: bool,
    capture_generation: u64,
    capture_started_ms: u64,
    capture_position_ms: u64,
}

impl Shared {
    fn capture_position(&self, now_ms: u64) -> u64 {
        self.capture_position_ms
            + if self.capturing {
                now_ms.saturating_sub(self.capture_started_ms)
            } else {
                0
            }
    }

    fn toggle_capture(&mut self, now_ms: u64) -> bool {
        if self.capturing {
            self.capture_position_ms = self.capture_position(now_ms);
            self.capturing = false;
        } else {
            self.capture_position_ms = self.capture_position_ms.max(
                self.notes
                    .iter()
                    .map(|note| note.onset_ms + note.duration_ms)
                    .max()
                    .unwrap_or(0),
            );
            self.capture_started_ms = now_ms;
            self.capturing = true;
        }
        self.capture_generation += 1;
        self.status = if self.capturing {
            "Recording MIDI prompt...".into()
        } else {
            "Recording stopped.".into()
        };
        self.capturing
    }

    fn clear_timeline(&mut self, now_ms: u64) {
        self.notes.clear();
        self.capture_position_ms = 0;
        self.capture_started_ms = now_ms;
        self.capture_generation += 1;
    }
}

#[derive(Clone, Debug, Default, Deserialize, PartialEq, Serialize)]
#[serde(default)]
struct AppConfig {
    midi_input: Option<String>,
    midi_output: Option<String>,
    soundfont: Option<String>,
}

fn config_path() -> PathBuf {
    if let Some(path) = std::env::var_os("XDG_CONFIG_HOME").filter(|path| !path.is_empty()) {
        return PathBuf::from(path).join("midi-autocomplete/config.toml");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".config/midi-autocomplete/config.toml");
    }
    PathBuf::from("config.toml")
}

fn load_config(path: &Path) -> Result<AppConfig, String> {
    match fs::read_to_string(path) {
        Ok(contents) => {
            toml::from_str(&contents).map_err(|error| format!("Invalid config: {error}"))
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(AppConfig::default()),
        Err(error) => Err(format!("Could not read config: {error}")),
    }
}

fn save_config(path: &Path, config: &AppConfig) -> Result<(), String> {
    let contents = toml::to_string_pretty(config).map_err(|error| error.to_string())?;
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("Could not create config directory: {error}"))?;
    }
    let temporary = path.with_extension("toml.tmp");
    fs::write(&temporary, contents).map_err(|error| format!("Could not write config: {error}"))?;
    fs::rename(&temporary, path).map_err(|error| format!("Could not save config: {error}"))
}

#[derive(Default)]
struct PlaybackControl {
    generation: AtomicU64,
    playing: AtomicBool,
    has_timeline: AtomicBool,
    wall_start_ms: AtomicU64,
    musical_start_ms: AtomicU64,
    musical_end_ms: AtomicU64,
    player: Mutex<Option<Arc<Player>>>,
}

impl PlaybackControl {
    fn begin(&self) -> u64 {
        self.has_timeline.store(false, Ordering::SeqCst);
        self.playing.store(true, Ordering::SeqCst);
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(player) = self.player.lock().unwrap().take() {
            player.stop();
        }
        generation
    }

    fn stop(&self) {
        self.has_timeline.store(false, Ordering::SeqCst);
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

    fn set_timeline(
        &self,
        generation: u64,
        wall_start_ms: u64,
        musical_start_ms: u64,
        musical_end_ms: u64,
    ) {
        if self.is_active(generation) {
            self.wall_start_ms.store(wall_start_ms, Ordering::SeqCst);
            self.musical_start_ms
                .store(musical_start_ms, Ordering::SeqCst);
            self.musical_end_ms.store(musical_end_ms, Ordering::SeqCst);
            self.has_timeline.store(true, Ordering::SeqCst);
        }
    }

    fn position(&self, now_ms: u64) -> Option<u64> {
        if !self.is_playing() || !self.has_timeline.load(Ordering::SeqCst) {
            return None;
        }
        Some(
            (self.musical_start_ms.load(Ordering::SeqCst)
                + now_ms.saturating_sub(self.wall_start_ms.load(Ordering::SeqCst)))
            .min(self.musical_end_ms.load(Ordering::SeqCst)),
        )
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
            self.has_timeline.store(false, Ordering::SeqCst);
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

fn preferred_index(names: &[String], preferred: Option<&str>) -> Option<u32> {
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
    let mut capture_generation = 0;
    input
        .connect(
            port,
            "midi-autocomplete-input",
            move |_, message, _| {
                let now = started.elapsed().as_millis() as u64;
                let (position, generation) = {
                    let state = shared.lock().unwrap();
                    (
                        state.capturing.then(|| state.capture_position(now)),
                        state.capture_generation,
                    )
                };
                if generation != capture_generation {
                    capture = Capture::new();
                    capture_generation = generation;
                }
                if let Some(position) = position {
                    capture.receive(message, position, &shared);
                }
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

fn connect_devices(
    input_index: u32,
    output_index: u32,
    started: Instant,
    shared: Arc<Mutex<Shared>>,
    input_slot: &Rc<RefCell<Option<MidiInputConnection<()>>>>,
    output_slot: &Arc<Mutex<Option<MidiOutputConnection>>>,
) -> Result<(), String> {
    let input = connect_input(input_index, started, shared)?;
    let output = connect_output(output_index)?;
    *input_slot.borrow_mut() = Some(input);
    *output_slot.lock().unwrap() = Some(output);
    Ok(())
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
const PIXELS_PER_MS: f64 = 0.08;

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
        let first = notes.iter().map(|note| note.onset_ms).min().unwrap();
        let end = notes
            .iter()
            .map(|note| note.onset_ms + note.duration_ms)
            .max()
            .unwrap();
        playback.set_timeline(generation, base + 100, first, end);
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
    let Some(end) = notes
        .iter()
        .map(|note| note.onset_ms + note.duration_ms)
        .max()
    else {
        shared.lock().unwrap().status = "Model generated no notes".into();
        playback.finish(generation);
        return;
    };
    playback.set_timeline(generation, live_base, musical_base, end);
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

fn timeline_bounds(notes: &[Note], playhead: Option<u64>) -> Option<(u64, u64)> {
    let start = notes.iter().map(|note| note.onset_ms).min().or(playhead)?;
    let end = notes
        .iter()
        .map(|note| note.onset_ms + note.duration_ms)
        .chain(playhead)
        .max()
        .unwrap_or(start);
    Some((start, end))
}

fn draw_roll(cr: &cairo::Context, width: i32, height: i32, notes: &[Note], playhead: Option<u64>) {
    cr.set_source_rgb(0.07, 0.08, 0.10);
    let _ = cr.paint();
    let start = timeline_bounds(notes, playhead).map_or(0, |bounds| bounds.0);
    for pitch in 21..=108 {
        let y = height as f64 * (108 - pitch) as f64 / 88.0;
        cr.set_source_rgba(1.0, 1.0, 1.0, if pitch % 12 == 0 { 0.10 } else { 0.025 });
        cr.rectangle(0.0, y, width as f64, 1.0);
        let _ = cr.fill();
    }
    for note in notes {
        let x = note.onset_ms.saturating_sub(start) as f64 * PIXELS_PER_MS;
        let w = (note.duration_ms as f64 * PIXELS_PER_MS).max(2.0);
        let y = height as f64 * (108_i32 - note.pitch as i32) as f64 / 88.0;
        if note.generated {
            cr.set_source_rgb(0.25, 0.80, 0.48);
        } else {
            cr.set_source_rgb(0.32, 0.58, 0.95);
        }
        cr.rectangle(x, y, w, (height as f64 / 88.0).max(3.0));
        let _ = cr.fill();
    }
    if let Some(playhead) = playhead {
        let x = playhead.saturating_sub(start) as f64 * PIXELS_PER_MS;
        cr.set_source_rgb(0.95, 0.18, 0.20);
        cr.rectangle(x, 0.0, 2.0, height as f64);
        let _ = cr.fill();
    }
}

fn build_ui(app: &Application) {
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
    let stop = Button::with_label("Stop");
    stop.set_visible(false);
    let browse = Button::with_label("Browse");
    let soundfont = Entry::builder()
        .placeholder_text("Select a .sf2 SoundFont")
        .hexpand(true)
        .build();
    if let Some(path) = &initial_config.soundfont {
        soundfont.set_text(path);
    }
    let config = Arc::new(Mutex::new(initial_config));
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
            None
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
    record.connect_clicked(move |_| {
        let mut state = state_for_record.lock().unwrap();
        if state.connected {
            state.toggle_capture(started.elapsed().as_millis() as u64);
        } else {
            state.status = "Connect MIDI devices before recording.".into();
        }
    });

    let state_for_clear = shared.clone();
    clear.connect_clicked(move |_| {
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

    let state_for_play = shared.clone();
    let output_for_play = output.clone();
    let playback_for_play = playback_control.clone();
    let soundfont_for_play = soundfont.clone();
    let config_for_play = config.clone();
    let path_for_play = config_path.clone();
    play.connect_clicked(move |_| {
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
    let timeline_for_timer = timeline.clone();
    let state_for_timer = shared.clone();
    let status_for_timer = status.clone();
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
            None
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
    devices.append(&refresh);
    devices.append(&record);

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
    fn config_round_trips() {
        let directory =
            std::env::temp_dir().join(format!("midi-autocomplete-{}", std::process::id()));
        let path = directory.join("config.toml");
        let expected = AppConfig {
            midi_input: Some("Piano In".into()),
            midi_output: Some("Piano Out".into()),
            soundfont: Some("/sounds/piano.sf2".into()),
        };
        save_config(&path, &expected).unwrap();
        assert_eq!(load_config(&path).unwrap(), expected);
        fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn device_selection_follows_names_after_refresh() {
        let names = vec!["Other".to_string(), "Piano".to_string()];
        assert_eq!(preferred_index(&names, Some("Piano")), Some(1));
        assert_eq!(preferred_index(&names, Some("Missing")), None);
    }

    #[test]
    fn playhead_maps_wall_time_to_musical_time() {
        let playback = PlaybackControl::default();
        let generation = playback.begin();
        playback.set_timeline(generation, 1_000, 500, 900);
        assert_eq!(playback.position(1_200), Some(700));
        assert_eq!(playback.position(2_000), Some(900));
        playback.finish(generation);
        assert_eq!(playback.position(2_000), None);
    }

    #[test]
    fn timeline_bounds_include_playhead() {
        let notes = [Note {
            pitch: 60,
            onset_ms: 500,
            duration_ms: 250,
            velocity: 80,
            generated: false,
        }];
        assert_eq!(timeline_bounds(&notes, Some(1_000)), Some((500, 1_000)));
    }

    #[test]
    fn recording_resumes_from_its_frozen_position() {
        let mut state = Shared::default();
        assert!(state.toggle_capture(100));
        assert_eq!(state.capture_position(250), 150);
        assert!(!state.toggle_capture(250));
        assert_eq!(state.capture_position(5_000), 150);
        assert!(state.toggle_capture(5_000));
        assert_eq!(state.capture_position(5_100), 250);
        assert_eq!(state.capture_generation, 3);
    }

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
