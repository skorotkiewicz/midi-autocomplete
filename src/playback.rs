use crate::state::{Note, Shared};
use midir::MidiOutputConnection;
use rodio::{DeviceSinkBuilder, Player, buffer::SamplesBuffer};
use rustysynth::{SoundFont, Synthesizer, SynthesizerSettings};
use std::fs::File;
use std::num::{NonZeroU16, NonZeroU32};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Default)]
pub(crate) struct PlaybackControl {
    generation: AtomicU64,
    playing: AtomicBool,
    has_timeline: AtomicBool,
    wall_start_ms: AtomicU64,
    musical_start_ms: AtomicU64,
    musical_end_ms: AtomicU64,
    player: Mutex<Option<Arc<Player>>>,
}

impl PlaybackControl {
    pub(crate) fn begin(&self) -> u64 {
        self.has_timeline.store(false, Ordering::SeqCst);
        self.playing.store(true, Ordering::SeqCst);
        let generation = self.generation.fetch_add(1, Ordering::SeqCst) + 1;
        if let Some(player) = self.player.lock().unwrap().take() {
            player.stop();
        }
        generation
    }

    pub(crate) fn stop(&self) {
        self.has_timeline.store(false, Ordering::SeqCst);
        self.playing.store(false, Ordering::SeqCst);
        self.generation.fetch_add(1, Ordering::SeqCst);
        if let Some(player) = self.player.lock().unwrap().take() {
            player.stop();
        }
    }

    pub(crate) fn is_active(&self, generation: u64) -> bool {
        self.generation.load(Ordering::SeqCst) == generation
    }

    pub(crate) fn is_playing(&self) -> bool {
        self.playing.load(Ordering::SeqCst)
    }

    pub(crate) fn set_timeline(
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

    pub(crate) fn position(&self, now_ms: u64) -> Option<u64> {
        if !self.is_playing() || !self.has_timeline.load(Ordering::SeqCst) {
            return None;
        }
        Some(
            (self.musical_start_ms.load(Ordering::SeqCst)
                + now_ms.saturating_sub(self.wall_start_ms.load(Ordering::SeqCst)))
            .min(self.musical_end_ms.load(Ordering::SeqCst)),
        )
    }

    pub(crate) fn set_player(&self, generation: u64, player: Arc<Player>) {
        let mut current = self.player.lock().unwrap();
        if self.is_active(generation) {
            *current = Some(player);
        } else {
            player.stop();
        }
    }

    pub(crate) fn finish(&self, generation: u64) {
        let mut current = self.player.lock().unwrap();
        if self.is_active(generation) {
            self.has_timeline.store(false, Ordering::SeqCst);
            self.playing.store(false, Ordering::SeqCst);
            current.take();
        }
    }
}

pub(crate) type MidiEvent = (u64, bool, u8, u8);

const SAMPLE_RATE: u32 = 48_000;
const MAX_PLAYBACK_MS: u64 = 5 * 60 * 1_000;

pub(crate) fn send_midi_events(
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

pub(crate) fn replay_events(notes: &[Note]) -> Result<Vec<MidiEvent>, String> {
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

pub(crate) fn render_soundfont(path: &Path, events: &[MidiEvent]) -> Result<Vec<f32>, String> {
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

pub(crate) fn replay(
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
        let mut device = match DeviceSinkBuilder::open_default_sink() {
            Ok(device) => device,
            Err(error) => {
                if playback.is_active(generation) {
                    shared.lock().unwrap().status = format!("Could not open audio output: {error}");
                }
                playback.finish(generation);
                return;
            }
        };
        device.log_on_drop(false);
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

pub(crate) fn play_generated(
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
