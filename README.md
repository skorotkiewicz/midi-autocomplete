![MIDI Autocomplete banner](assets/banner.svg)

# MIDI Autocomplete

A GTK4 desktop piano-continuation app based on Simon Edwardsson's [MIDI autocomplete write-up](https://simedw.com/2026/08/20/midi-autocomplete/).

## Run

Requirements: GTK4, ALSA development headers on Linux, Rust, and [uv](https://docs.astral.sh/uv/).

```sh
# Train a useful checkpoint first. See midilm/README.md.
cargo run --release
```

1. Open **Settings**, then select a MIDI keyboard input and output. Click **Refresh** after you connect new hardware.
2. In **Settings**, select a local checkpoint, performance BPM, and `.sf2` SoundFont.
3. Click **Connect**, then click **Rec** and play a prompt.
4. Use **Auto** or **Explicit** generation.

**Auto** is the default mode. After 800 ms without a pressed or sustained note, the app stops recording and generates a continuation. It renders the continuation with the selected SoundFont before it sends audio or MIDI. Recording resumes at the end of the continuation.

**Explicit** waits for **Autocomplete**. It can generate from the recorded prompt or from silence. Generated notes stay silent until you click **Play** and SoundFont rendering finishes.

The red playhead follows recording and playback on the scrollable timeline. Click the timeline to seek. **Pause** freezes playback at the playhead, and **Resume** continues from that position. **Stop** keeps the position for the next Autocomplete request. **Clear** resets the notes and playhead.

Play renders the SoundFont through the computer's default audio device and sends the same notes to the selected MIDI output. Blue notes are MIDI input. Green notes come from the model. The app records velocity and duration, including sustain-pedal time.

**File → Import MIDI** replaces the timeline with notes from a standard `.mid` or `.midi` file. The app merges parallel tracks, applies tempo changes, and folds sustain into note duration. Imported notes are blue. Sequential multi-song files and SMPTE timing are not supported.

**File → Export MIDI** writes all timeline notes to one piano track at the selected BPM. Standard MIDI does not store whether this app or the model created a note, so that distinction is not preserved after import.

## Configuration

The app saves the MIDI device selections, model checkpoint, and SoundFont path automatically. On startup it restores them and reconnects when both saved MIDI devices are available. If no model is saved, it uses `midilm/checkpoints/medium.pt`.

```text
~/.config/midi-autocomplete/config.toml
```

`$XDG_CONFIG_HOME` replaces `~/.config` when set.

```toml
midi_input = "Digital Piano MIDI 1"
midi_output = "Digital Piano MIDI 1"
model = "midilm/checkpoints/medium.pt"
soundfont = "/home/user/sounds/piano.sf2"
```
