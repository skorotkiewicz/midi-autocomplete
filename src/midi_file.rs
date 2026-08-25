use crate::state::Note;
use midly::{
    Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind,
    num::{u4, u7, u15, u24, u28},
};
use std::fs;
use std::path::Path;

const TICKS_PER_QUARTER: u16 = 24;
const DEFAULT_TEMPO_US: u32 = 500_000;
const MAX_MIDI_FILE_BYTES: u64 = 64 * 1024 * 1024;

pub(crate) struct ImportedMidi {
    pub(crate) notes: Vec<Note>,
    pub(crate) bpm: f64,
}

#[derive(Clone, Copy)]
struct HeldNote {
    onset_ms: u64,
    velocity: u8,
    released: bool,
}

#[derive(Clone, Copy)]
enum AbsoluteKind {
    Tempo(u32),
    Midi { channel: u8, message: MidiMessage },
}

#[derive(Clone, Copy)]
struct AbsoluteEvent {
    tick: u64,
    priority: u8,
    sequence: usize,
    kind: AbsoluteKind,
}

pub(crate) fn import_midi(path: &Path) -> Result<ImportedMidi, String> {
    let metadata =
        fs::metadata(path).map_err(|error| format!("Could not read MIDI file: {error}"))?;
    if metadata.len() > MAX_MIDI_FILE_BYTES {
        return Err("MIDI file is larger than 64 MB".into());
    }
    let bytes = fs::read(path).map_err(|error| format!("Could not read MIDI file: {error}"))?;
    decode_midi(&bytes)
}

pub(crate) fn decode_midi(bytes: &[u8]) -> Result<ImportedMidi, String> {
    let smf = Smf::parse(bytes).map_err(|error| format!("Invalid MIDI file: {error}"))?;
    if smf.header.format == Format::Sequential {
        return Err("Sequential multi-song MIDI files are not supported".into());
    }
    let ticks_per_quarter = match smf.header.timing {
        Timing::Metrical(ticks) if ticks.as_int() > 0 => u64::from(ticks.as_int()),
        Timing::Metrical(_) => return Err("MIDI file has zero ticks per quarter note".into()),
        Timing::Timecode(_, _) => return Err("SMPTE-timed MIDI files are not supported".into()),
    };

    let mut events = Vec::new();
    let mut sequence = 0;
    let mut final_tick = 0;
    for track in &smf.tracks {
        let mut tick = 0_u64;
        for event in track {
            tick = tick.saturating_add(u64::from(event.delta.as_int()));
            final_tick = final_tick.max(tick);
            let kind = match event.kind {
                TrackEventKind::Meta(MetaMessage::Tempo(tempo)) => {
                    Some((0, AbsoluteKind::Tempo(tempo.as_int())))
                }
                TrackEventKind::Midi { channel, message }
                    if matches!(
                        message,
                        MidiMessage::NoteOff { .. } | MidiMessage::NoteOn { .. }
                    ) || matches!(
                        message,
                        MidiMessage::Controller { controller, .. }
                            if controller.as_int() == 64
                    ) =>
                {
                    Some((
                        1,
                        AbsoluteKind::Midi {
                            channel: channel.as_int(),
                            message,
                        },
                    ))
                }
                _ => None,
            };
            if let Some((priority, kind)) = kind {
                events.push(AbsoluteEvent {
                    tick,
                    priority,
                    sequence,
                    kind,
                });
                sequence += 1;
            }
        }
    }
    events.sort_by_key(|event| (event.tick, event.priority, event.sequence));

    let mut notes = Vec::new();
    let mut held = [None; 16 * 128];
    let mut sustain = [false; 16];
    let mut tempo_us = DEFAULT_TEMPO_US;
    let mut first_tempo_us = None;
    let mut previous_tick = 0;
    let mut elapsed_us = 0_u128;
    for event in events {
        let delta_ticks = event.tick.saturating_sub(previous_tick);
        elapsed_us = elapsed_us
            .checked_add(
                u128::from(delta_ticks) * u128::from(tempo_us) / u128::from(ticks_per_quarter),
            )
            .ok_or("MIDI timeline is too long")?;
        let now_ms = u64::try_from(elapsed_us / 1_000).map_err(|_| "MIDI timeline is too long")?;
        previous_tick = event.tick;
        match event.kind {
            AbsoluteKind::Tempo(value) if value > 0 => {
                first_tempo_us.get_or_insert(value);
                tempo_us = value;
            }
            AbsoluteKind::Tempo(_) => return Err("MIDI file contains a zero tempo".into()),
            AbsoluteKind::Midi { channel, message } => apply_midi_event(
                message,
                channel,
                now_ms,
                &mut held,
                &mut sustain,
                &mut notes,
            ),
        }
    }

    let remaining_ticks = final_tick.saturating_sub(previous_tick);
    elapsed_us = elapsed_us
        .checked_add(
            u128::from(remaining_ticks) * u128::from(tempo_us) / u128::from(ticks_per_quarter),
        )
        .ok_or("MIDI timeline is too long")?;
    let end_ms = u64::try_from(elapsed_us / 1_000).map_err(|_| "MIDI timeline is too long")?;
    for index in 0..held.len() {
        finish_note(index, end_ms, &mut held, &mut notes);
    }
    if notes.is_empty() {
        return Err("MIDI file contains no notes".into());
    }
    notes.sort_by_key(|note| (note.onset_ms, note.pitch));
    let tempo = first_tempo_us.unwrap_or(DEFAULT_TEMPO_US);
    Ok(ImportedMidi {
        notes,
        bpm: 60_000_000.0 / f64::from(tempo),
    })
}

fn apply_midi_event(
    message: MidiMessage,
    channel: u8,
    now_ms: u64,
    held: &mut [Option<HeldNote>; 16 * 128],
    sustain: &mut [bool; 16],
    notes: &mut Vec<Note>,
) {
    match message {
        MidiMessage::NoteOn { key, vel } if vel.as_int() > 0 => {
            let index = usize::from(channel) * 128 + usize::from(key.as_int());
            finish_note(index, now_ms, held, notes);
            held[index] = Some(HeldNote {
                onset_ms: now_ms,
                velocity: vel.as_int(),
                released: false,
            });
        }
        MidiMessage::NoteOff { key, .. } | MidiMessage::NoteOn { key, .. } => {
            let index = usize::from(channel) * 128 + usize::from(key.as_int());
            if sustain[usize::from(channel)] {
                if let Some(note) = held[index].as_mut() {
                    note.released = true;
                }
            } else {
                finish_note(index, now_ms, held, notes);
            }
        }
        MidiMessage::Controller { controller, value } if controller.as_int() == 64 => {
            let channel = usize::from(channel);
            let was_down = sustain[channel];
            sustain[channel] = value.as_int() >= 64;
            if was_down && !sustain[channel] {
                for pitch in 0..128 {
                    let index = channel * 128 + pitch;
                    if held[index].is_some_and(|note| note.released) {
                        finish_note(index, now_ms, held, notes);
                    }
                }
            }
        }
        _ => {}
    }
}

fn finish_note(
    index: usize,
    end_ms: u64,
    held: &mut [Option<HeldNote>; 16 * 128],
    notes: &mut Vec<Note>,
) {
    if let Some(note) = held[index].take() {
        notes.push(Note {
            pitch: (index % 128) as u8,
            onset_ms: note.onset_ms,
            duration_ms: end_ms.saturating_sub(note.onset_ms).max(1),
            velocity: note.velocity,
            generated: false,
        });
    }
}

pub(crate) fn export_midi(path: &Path, notes: &[Note], bpm: f64) -> Result<(), String> {
    let bytes = encode_midi(notes, bpm)?;
    fs::write(path, bytes).map_err(|error| format!("Could not write MIDI file: {error}"))
}

pub(crate) fn encode_midi(notes: &[Note], bpm: f64) -> Result<Vec<u8>, String> {
    if notes.is_empty() {
        return Err("Nothing to export".into());
    }
    if !bpm.is_finite() || bpm <= 0.0 {
        return Err("BPM must be greater than zero".into());
    }
    let tempo_us = (60_000_000.0 / bpm).round();
    if !(1.0..=f64::from(0x00ff_ffff)).contains(&tempo_us) {
        return Err("BPM is outside the MIDI tempo range".into());
    }
    let step_ms = 60_000.0 / bpm / f64::from(TICKS_PER_QUARTER);
    let mut events = vec![(
        0_u64,
        0_u8,
        TrackEventKind::Meta(MetaMessage::Tempo(u24::new(tempo_us as u32))),
    )];
    for note in notes {
        if note.pitch > 127 || note.velocity > 127 {
            return Err("A note is outside the MIDI range".into());
        }
        let onset = (note.onset_ms as f64 / step_ms).round() as u64;
        let duration = (note.duration_ms as f64 / step_ms).round().max(1.0) as u64;
        events.push((
            onset,
            2,
            TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::NoteOn {
                    key: u7::new(note.pitch),
                    vel: u7::new(note.velocity),
                },
            },
        ));
        events.push((
            onset.saturating_add(duration),
            1,
            TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::NoteOff {
                    key: u7::new(note.pitch),
                    vel: u7::new(0),
                },
            },
        ));
    }
    events.sort_by_key(|(tick, priority, _)| (*tick, *priority));

    let mut previous_tick = 0;
    let mut track = Vec::with_capacity(events.len() + 1);
    for (tick, _, kind) in events {
        let delta = tick.saturating_sub(previous_tick);
        let delta = u32::try_from(delta)
            .ok()
            .and_then(u28::try_from)
            .ok_or("MIDI event delta exceeds the file format limit")?;
        track.push(TrackEvent { delta, kind });
        previous_tick = tick;
    }
    track.push(TrackEvent {
        delta: u28::new(0),
        kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
    });
    let smf = Smf {
        header: Header::new(
            Format::SingleTrack,
            Timing::Metrical(u15::new(TICKS_PER_QUARTER)),
        ),
        tracks: vec![track],
    };
    let mut bytes = Vec::new();
    smf.write_std(&mut bytes)
        .map_err(|error| format!("Could not encode MIDI file: {error}"))?;
    Ok(bytes)
}
