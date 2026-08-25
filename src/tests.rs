use crate::config::{AppConfig, config_path, load_config, save_config};
use crate::midi::{connect_output, midi_inputs, midi_outputs};
use crate::model::{prompt, repo_root, resolve_model_path};
use crate::playback::{
    PlaybackControl, add_generated, render_soundfont, replay_events_from, send_midi_events,
};
use crate::state::{Note, Shared};
use crate::timeline::timeline_bounds;
use crate::ui::{auto_generation_due, preferred_index};
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
    };
    save_config(&path, &expected).unwrap();
    assert_eq!(load_config(&path).unwrap(), expected);
    fs::remove_dir_all(directory).unwrap();
}

#[test]
fn auto_generation_waits_for_a_complete_idle_pause() {
    assert!(!auto_generation_due(799, 0, false, 1, 0));
    assert!(!auto_generation_due(800, 0, true, 1, 0));
    assert!(!auto_generation_due(800, 0, false, 1, 1));
    assert!(auto_generation_due(800, 0, false, 1, 0));
}

#[test]
fn device_selection_follows_names_after_refresh() {
    let names = vec!["Other".to_string(), "Piano".to_string()];
    assert_eq!(preferred_index(&names, Some("Piano")), Some(1));
    assert_eq!(preferred_index(&names, Some("Missing")), None);
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
    assert!(playback.is_paused());
    assert!(!playback.is_playing());

    assert!(!playback.seek(1_500));
    assert_eq!(playback.cursor(), Some(1_500));
    assert!(playback.is_paused());
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
    assert_eq!(state.capture_position(250), 150);
    assert!(!state.toggle_capture(250, None));
    assert_eq!(state.capture_position(5_000), 150);
    assert!(state.toggle_capture(5_000, None));
    assert_eq!(state.capture_position(5_100), 250);
    assert_eq!(state.capture_generation, 3);
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
    assert_eq!(state.capture_position(200), 350);
}

#[test]
fn generated_notes_wait_for_soundfont_playback() {
    let shared = Arc::new(Mutex::new(Shared::default()));
    add_generated(vec![(64, 0, 12, 90)], 120.0, None, shared.clone());

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
    add_generated(vec![(64, 0, 12, 90)], 120.0, Some(500), shared.clone());

    let state = shared.lock().unwrap();
    assert_eq!(state.notes.len(), 2);
    assert_eq!(state.notes[1].pitch, 64);
    assert_eq!(state.notes[1].onset_ms, 500);
}

#[test]
fn stop_cancels_scheduled_midi_without_waiting() {
    let playback = Arc::new(PlaybackControl::default());
    let generation = playback.begin_rendering();
    assert!(playback.start_playback(generation));
    assert!(playback.is_playing());
    playback.stop(0);
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
