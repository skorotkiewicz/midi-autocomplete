use crate::config::{AppConfig, config_path, load_config, save_config};
use crate::midi::{connect_output, midi_inputs, midi_outputs};
use crate::midi_file::{decode_midi, encode_midi};
use crate::model::{parse_generated_notes, prompt, repo_root, resolve_model_path};
use crate::playback::{
    GeneratedUpdate, PlaybackControl, add_generated, render_soundfont, replay_events_from,
    send_midi_events,
};
use crate::state::{Note, Shared};
use crate::timeline::{
    playhead_overlay_x, scroll_for_playhead, timeline_bounds, timeline_content_width,
};
use crate::ui::{auto_generation_due, preferred_index};
use midly::{
    Format, Header, MetaMessage, MidiMessage, Smf, Timing, TrackEvent, TrackEventKind,
    num::{u4, u7, u15, u28},
};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[test]
fn config_round_trips() {
    let directory = std::env::temp_dir().join(format!("midi-autocomplete-{}", std::process::id()));
    let path = directory.join("config.toml");
    let expected = AppConfig {
        midi_input: Some("Piano In".into()),
        midi_output: Some("Piano Out".into()),
        model: Some("/models/medium.pt".into()),
        soundfont: Some("/sounds/piano.sf2".into()),
        auto_pause_ms: 1_200,
    };
    save_config(&path, &expected).unwrap();
    assert_eq!(load_config(&path).unwrap(), expected);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn auto_generation_waits_for_a_complete_idle_pause() {
    assert!(!auto_generation_due(800, 0, 800, false, false, 1, 0));
    assert!(!auto_generation_due(799, 0, 800, true, false, 1, 0));
    assert!(!auto_generation_due(800, 0, 800, true, true, 1, 0));
    assert!(!auto_generation_due(800, 0, 800, true, false, 1, 1));
    assert!(auto_generation_due(800, 0, 800, true, false, 1, 0));
    assert!(!auto_generation_due(1_199, 0, 1_200, true, false, 1, 0));
    assert!(auto_generation_due(1_200, 0, 1_200, true, false, 1, 0));
}

#[test]
fn device_selection_follows_names_after_refresh() {
    let names = vec!["Other".to_string(), "Piano".to_string()];
    assert_eq!(preferred_index(&names, Some("Piano")), Some(1));
    assert_eq!(preferred_index(&names, Some("Missing")), None);
}

#[test]
fn midi_file_round_trips_timeline_notes() {
    let bytes = encode_midi(
        &[Note {
            pitch: 60,
            onset_ms: 500,
            duration_ms: 250,
            velocity: 80,
            generated: true,
        }],
        120.0,
    )
    .unwrap();
    let imported = decode_midi(&bytes).unwrap();

    assert_eq!(imported.bpm, 120.0);
    assert_eq!(imported.notes.len(), 1);
    assert_eq!(imported.notes[0].pitch, 60);
    assert_eq!(imported.notes[0].onset_ms, 500);
    assert_eq!(imported.notes[0].duration_ms, 250);
    assert_eq!(imported.notes[0].velocity, 80);
    assert!(!imported.notes[0].generated);
}

#[test]
fn midi_import_folds_sustain_into_note_duration() {
    let track = vec![
        TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::NoteOn {
                    key: u7::new(60),
                    vel: u7::new(80),
                },
            },
        },
        TrackEvent {
            delta: u28::new(24),
            kind: TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::Controller {
                    controller: u7::new(64),
                    value: u7::new(127),
                },
            },
        },
        TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::NoteOff {
                    key: u7::new(60),
                    vel: u7::new(0),
                },
            },
        },
        TrackEvent {
            delta: u28::new(24),
            kind: TrackEventKind::Midi {
                channel: u4::new(0),
                message: MidiMessage::Controller {
                    controller: u7::new(64),
                    value: u7::new(0),
                },
            },
        },
        TrackEvent {
            delta: u28::new(0),
            kind: TrackEventKind::Meta(MetaMessage::EndOfTrack),
        },
    ];
    let smf = Smf {
        header: Header::new(Format::SingleTrack, Timing::Metrical(u15::new(24))),
        tracks: vec![track],
    };
    let mut bytes = Vec::new();
    smf.write_std(&mut bytes).unwrap();

    let imported = decode_midi(&bytes).unwrap();
    assert_eq!(imported.notes.len(), 1);
    assert_eq!(imported.notes[0].duration_ms, 1_000);
}

#[test]
fn model_output_rejects_values_outside_midi_range() {
    assert!(parse_generated_notes("128,0,12,80").is_err());
    assert!(parse_generated_notes("60,0,12,128").is_err());
    assert_eq!(
        parse_generated_notes("60,0,12,80").unwrap(),
        vec![(60, 0, 12, 80)]
    );
}

#[test]
fn model_path_passes_urls_through_and_resolves_locals_from_repo_root() {
    let url = "https://huggingface.co/Grizzlykw/midilm/resolve/main/medium.resume.pt?download=true";
    assert_eq!(resolve_model_path(Path::new(url)), PathBuf::from(url));
    let missing = Path::new("midilm/checkpoints/nope.pt");
    assert_eq!(
        resolve_model_path(missing),
        repo_root().join("midilm/checkpoints/nope.pt")
    );
}

#[test]
fn playhead_maps_wall_time_to_musical_time() {
    let playback = PlaybackControl::default();
    let generation = playback.begin_rendering();
    assert!(playback.start_playback(generation));
    playback.set_timeline(generation, 1_000, 500, 900);
    assert_eq!(playback.position(1_200), Some(700));
    assert_eq!(playback.position(2_000), Some(900));
    playback.finish(generation);
    assert_eq!(playback.position(2_000), None);
    assert!(!playback.is_active(generation));
}

#[test]
fn soundfont_rendering_is_not_playback() {
    let playback = PlaybackControl::default();
    let generation = playback.begin_rendering();
    assert!(playback.is_rendering());
    assert!(!playback.is_playing());

    assert!(playback.start_playback(generation));
    assert!(!playback.is_rendering());
    assert!(playback.is_playing());
}

#[test]
fn pause_and_seek_preserve_the_cursor() {
    let playback = PlaybackControl::default();
    let generation = playback.begin_rendering();
    assert!(playback.start_playback(generation));
    playback.set_timeline(generation, 1_000, 500, 2_000);

    assert!(playback.pause(1_250));
    assert_eq!(playback.cursor(), Some(750));
    assert_eq!(playback.playback_start(), Some(750));
    assert!(playback.is_paused());
    assert!(!playback.is_playing());

    assert!(!playback.seek(1_500));
    assert_eq!(playback.cursor(), Some(1_500));
    assert_eq!(playback.playback_start(), Some(1_500));
    assert!(playback.is_paused());
}

#[test]
fn playhead_stays_at_twenty_percent_after_the_opening() {
    assert_eq!(scroll_for_playhead(100.0, 1_000.0), 0.0);
    assert_eq!(scroll_for_playhead(200.0, 1_000.0), 0.0);
    assert_eq!(scroll_for_playhead(500.0, 1_000.0), 300.0);

    let content_width = timeline_content_width(0, 12_500, 1_000);
    let maximum_scroll = content_width - 1_000;
    assert_eq!(1_000 - maximum_scroll, 200);

    assert_eq!(playhead_overlay_x(&[], 1_000, 1_000), 0.0);
    let notes = [Note {
        pitch: 60,
        onset_ms: 0,
        duration_ms: 20_000,
        velocity: 80,
        generated: false,
    }];
    assert_eq!(playhead_overlay_x(&notes, 1_000, 1_000), 80.0);
    assert_eq!(playhead_overlay_x(&notes, 10_000, 1_000), 200.0);
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
    assert!(state.toggle_capture(100, None));
    assert_eq!(state.capture_position(250), 0);
    state.capture_started_ms = 250;
    state.capture_has_input = true;
    assert_eq!(state.capture_position(400), 150);

    assert!(!state.toggle_capture(400, None));
    assert_eq!(state.capture_position(5_000), 150);
    assert!(state.toggle_capture(5_000, None));
    assert_eq!(state.capture_position(5_100), 150);
    state.capture_started_ms = 5_100;
    state.capture_has_input = true;
    assert_eq!(state.capture_position(5_200), 250);
    assert_eq!(state.capture_generation, 3);
}

#[test]
fn every_recording_session_waits_for_fresh_input() {
    let mut state = Shared::default();
    assert!(state.toggle_capture(0, Some(500)));
    assert_eq!(state.capture_position(1_000), 500);

    state.capture_started_ms = 1_000;
    state.capture_has_input = true;
    assert_eq!(state.capture_position(1_100), 600);

    assert!(!state.toggle_capture(1_100, None));
    assert!(state.toggle_capture(1_200, None));
    assert!(!state.capture_has_input);
    assert_eq!(state.capture_position(2_000), 600);
}

#[test]
fn recording_starts_from_the_selected_timeline_position() {
    let mut state = Shared {
        notes: vec![Note {
            pitch: 60,
            onset_ms: 1_000,
            duration_ms: 500,
            velocity: 80,
            generated: false,
        }],
        ..Default::default()
    };

    assert!(state.toggle_capture(100, Some(250)));
    assert_eq!(state.capture_position(200), 250);
    state.capture_started_ms = 200;
    state.capture_has_input = true;
    assert_eq!(state.capture_position(300), 350);
}

#[test]
fn generated_notes_wait_for_soundfont_playback() {
    let shared = Arc::new(Mutex::new(Shared::default()));
    assert_eq!(
        add_generated(vec![(64, 0, 12, 90)], 120.0, None, 0, shared.clone(),),
        GeneratedUpdate::Applied(Some(0))
    );

    let state = shared.lock().unwrap();
    assert_eq!(state.notes.len(), 1);
    assert!(state.notes[0].generated);
    assert_eq!(
        state.status,
        "Generated 1 notes. Click Play to render them."
    );
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
        replay_events_from(&notes, 500).unwrap(),
        vec![
            (100, true, 60, 80),
            (350, false, 60, 0),
            (600, true, 64, 90),
            (700, false, 64, 0),
        ]
    );
}

#[test]
fn replay_from_cursor_clips_notes_already_in_progress() {
    let notes = [
        Note {
            pitch: 60,
            onset_ms: 500,
            duration_ms: 500,
            velocity: 80,
            generated: false,
        },
        Note {
            pitch: 64,
            onset_ms: 1_200,
            duration_ms: 100,
            velocity: 90,
            generated: true,
        },
    ];
    assert_eq!(
        replay_events_from(&notes, 750).unwrap(),
        vec![
            (100, true, 60, 80),
            (350, false, 60, 0),
            (550, true, 64, 90),
            (650, false, 64, 0),
        ]
    );
}

#[test]
fn autocomplete_replaces_generated_notes_after_cursor() {
    let shared = Arc::new(Mutex::new(Shared {
        notes: vec![
            Note {
                pitch: 60,
                onset_ms: 0,
                duration_ms: 100,
                velocity: 80,
                generated: false,
            },
            Note {
                pitch: 62,
                onset_ms: 1_000,
                duration_ms: 100,
                velocity: 80,
                generated: true,
            },
        ],
        ..Default::default()
    }));
    assert_eq!(
        add_generated(vec![(64, 0, 12, 90)], 120.0, Some(500), 0, shared.clone(),),
        GeneratedUpdate::Applied(Some(500))
    );

    let state = shared.lock().unwrap();
    assert_eq!(state.notes.len(), 2);
    assert_eq!(state.notes[1].pitch, 64);
    assert_eq!(state.notes[1].onset_ms, 500);
}

#[test]
fn outdated_generation_cannot_reappear_after_timeline_changes() {
    let shared = Arc::new(Mutex::new(Shared::default()));
    shared.lock().unwrap().invalidate_generation();

    assert_eq!(
        add_generated(vec![(64, 0, 12, 90)], 120.0, None, 0, shared.clone(),),
        GeneratedUpdate::Stale
    );
    assert!(shared.lock().unwrap().notes.is_empty());
}

#[test]
fn stop_cancels_scheduled_midi_without_waiting() {
    let playback = Arc::new(PlaybackControl::default());
    let generation = playback.begin_rendering();
    assert!(playback.start_playback(generation));
    playback.set_timeline(generation, 0, 500, 900);
    assert!(playback.is_playing());
    playback.stop(100);
    assert!(!playback.is_playing());
    assert_eq!(playback.cursor(), Some(600));
    assert_eq!(playback.playback_start(), None);
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
fn prompt_stops_at_the_cursor() {
    let notes = [
        Note {
            pitch: 60,
            onset_ms: 0,
            duration_ms: 250,
            velocity: 80,
            generated: false,
        },
        Note {
            pitch: 64,
            onset_ms: 1_000,
            duration_ms: 250,
            velocity: 90,
            generated: false,
        },
    ];
    assert_eq!(prompt(&notes, 120.0, Some(500)), "60,0,12,80");
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
    assert_eq!(prompt(&notes, 120.0, None), "60,0,12,80");
}

// Hardware-in-the-loop integration tests. Run with `cargo test -- --ignored`.
// These need the actual RockJam board connected (per config.toml) and only
// touch it in read-only ways that don't leave stuck notes: enumerate ports,
// open an output connection, and render a real SoundFont to valid samples.
mod hardware {
    use super::*;

    fn config() -> AppConfig {
        load_config(&config_path()).expect("config.toml should exist")
    }

    fn board_name(role: &str) -> Option<String> {
        let config = config();
        let name = if role == "input" {
            config.midi_input
        } else {
            config.midi_output
        };
        name.filter(|name| !name.trim().is_empty())
    }

    #[test]
    #[ignore]
    fn input_port_matches_config() {
        let name = board_name("input").expect("midi_input set in config");
        let present = midi_inputs().iter().any(|candidate| candidate == &name);
        assert!(
            present,
            "input device '{name}' should appear in midi_inputs()"
        );
    }

    #[test]
    #[ignore]
    fn output_port_matches_config() {
        let name = board_name("output").expect("midi_output set in config");
        let present = midi_outputs().iter().any(|candidate| candidate == &name);
        assert!(
            present,
            "output device '{name}' should appear in midi_outputs()"
        );
    }

    #[test]
    #[ignore]
    fn output_port_opens_connection() {
        let name = board_name("output").expect("midi_output set in config");
        let names = midi_outputs();
        let index = names
            .iter()
            .position(|candidate| candidate == &name)
            .expect("output device present");
        let connection =
            connect_output(index as u32).expect("should open a connection to the RockJam output");
        drop(connection); // no notes sent; just verify the port is connectable
    }

    #[test]
    #[ignore]
    fn soundfont_renders_note_to_samples() {
        let path = PathBuf::from(config().soundfont.expect("soundfont set in config"));
        assert!(
            path.exists(),
            "SoundFont file should exist: {}",
            path.display()
        );
        let events = vec![(0, true, 60, 100), (500, false, 60, 0)];
        let samples =
            render_soundfont(&path, &events).expect("should render the SoundFont without error");
        // ~500ms at 48kHz stereo = ~48k samples; real note should be audible (non-silent)
        assert!(
            samples.len() > 40_000,
            "samples should cover the note duration"
        );
        let energy = samples
            .iter()
            .map(|sample| sample.abs())
            .fold(0.0_f64, |acc, sample| acc + sample as f64);
        assert!(
            energy > 0.0,
            "a real note should produce nonzero audio energy"
        );
    }
}
