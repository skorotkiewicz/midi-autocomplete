# MIDI Autocomplete

A GTK4 desktop piano-continuation app based on Simon Edwardsson's [MIDI autocomplete write-up](https://simedw.com/2026/08/20/midi-autocomplete/).

## Run

Requirements: GTK4, ALSA development headers on Linux, Rust, and [uv](https://docs.astral.sh/uv/).

```sh
# Train a useful checkpoint first. See midilm/README.md.
cargo run --release
```

1. Select a MIDI keyboard input and, optionally, a synth or keyboard output.
2. Click **Connect** and play a prompt.
3. Set the checkpoint path and performance BPM, then click **Autocomplete**.
4. Choose a `.sf2` SoundFont and click **Play** to replay the full piano roll.

Play renders the SoundFont through the computer's default audio device and simultaneously sends notes to the connected MIDI output. Blue piano-roll notes are input and green notes come from the model. The app records note duration and folds sustain-pedal time into it before inference.
