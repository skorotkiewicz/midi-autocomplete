use crate::state::{Note, Shared};
use midir::{Ignore, MidiInput, MidiInputConnection, MidiOutput, MidiOutputConnection};
use midly::{MidiMessage, live::LiveEvent};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Instant;

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
            shared.lock().unwrap().add_input_note(Note {
                pitch,
                onset_ms: note.onset_ms,
                duration_ms: end_ms.saturating_sub(note.onset_ms).max(1),
                velocity: note.velocity,
                generated: false,
            });
        }
    }

    fn receive(&mut self, bytes: &[u8], now_ms: u64, shared: &Arc<Mutex<Shared>>) -> bool {
        let Ok(LiveEvent::Midi { message, .. }) = LiveEvent::parse(bytes) else {
            return false;
        };
        // ponytail: piano input treats all MIDI channels as one; split by channel if multi-instrument input matters.
        match message {
            MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
                let pitch = key.as_int();
                self.finish(pitch, now_ms, shared);
                self.held[pitch as usize] = Some(HeldNote {
                    onset_ms: now_ms,
                    velocity: vel.as_int(),
                    released: false,
                });
            }
            MidiMessage::NoteOff { key, .. } | MidiMessage::NoteOn { key, .. } => {
                let pitch = key.as_int();
                if self.sustain {
                    if let Some(note) = self.held[pitch as usize].as_mut() {
                        note.released = true;
                    }
                } else {
                    self.finish(pitch, now_ms, shared);
                }
            }
            MidiMessage::Controller { controller, value } if controller.as_int() == 64 => {
                let was_down = self.sustain;
                self.sustain = value.as_int() >= 64;
                if was_down && !self.sustain {
                    for pitch in 0..128 {
                        if self.held[pitch].is_some_and(|note| note.released) {
                            self.finish(pitch as u8, now_ms, shared);
                        }
                    }
                }
            }
            _ => return false,
        }
        true
    }
}

pub(crate) fn midi_inputs() -> Vec<String> {
    let Ok(input) = MidiInput::new("midi-autocomplete-list") else {
        return Vec::new();
    };
    input
        .ports()
        .iter()
        .filter_map(|port| input.port_name(port).ok())
        .collect()
}

pub(crate) fn midi_outputs() -> Vec<String> {
    let Ok(output) = MidiOutput::new("midi-autocomplete-list") else {
        return Vec::new();
    };
    output
        .ports()
        .iter()
        .filter_map(|port| output.port_name(port).ok())
        .collect()
}

/// Send all-notes-off + reset to the connected output so notes don't hang when
/// the connection is dropped (connect/refresh swaps). Bluetooth MIDI in
/// particular can leave tails if a session closes during a held note.
pub(crate) fn silence_output(output: &Arc<Mutex<Option<MidiOutputConnection>>>) {
    if let Some(connection) = output.lock().unwrap().as_mut() {
        let _ = connection.send(&[0xb0, 120, 0]);
        let _ = connection.send(&[0xb0, 123, 0]);
    }
}

pub(crate) fn connect_input(
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
                if let Some(position) = position
                    && capture.receive(message, position, &shared)
                {
                    let mut state = shared.lock().unwrap();
                    state.input_active = capture.held.iter().any(Option::is_some);
                    state.last_input_ms = now;
                }
            },
            (),
        )
        .map_err(|error| error.to_string())
}

pub(crate) fn connect_output(selected: u32) -> Result<MidiOutputConnection, String> {
    let output = MidiOutput::new("midi-autocomplete-output").map_err(|error| error.to_string())?;
    let ports = output.ports();
    let port = ports.get(selected as usize).ok_or("Select a MIDI output")?;
    output
        .connect(port, "midi-autocomplete-output")
        .map_err(|error| error.to_string())
}

pub(crate) fn connect_devices(
    input_index: u32,
    output_index: u32,
    started: Instant,
    shared: Arc<Mutex<Shared>>,
    input_slot: &Rc<RefCell<Option<MidiInputConnection<()>>>>,
    output_slot: &Arc<Mutex<Option<MidiOutputConnection>>>,
) -> Result<(), String> {
    let input = connect_input(input_index, started, shared)?;
    let output = connect_output(output_index)?;
    silence_output(output_slot);
    *input_slot.borrow_mut() = Some(input);
    *output_slot.lock().unwrap() = Some(output);
    Ok(())
}
