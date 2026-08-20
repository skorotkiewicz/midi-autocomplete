# MIDI Autocomplete

A GTK4 desktop piano-continuation app based on Simon Edwardsson's [MIDI autocomplete write-up](https://simedw.com/2026/08/20/midi-autocomplete/).

## Run

Requirements: GTK4, ALSA development headers on Linux, Rust, and [uv](https://docs.astral.sh/uv/).

```sh
# Train a useful checkpoint first. See midilm/README.md.
cargo run --release
```

1. Select a MIDI keyboard input and a synth or keyboard output.
2. Click **Connect** and play a prompt.
3. Set the checkpoint path and performance BPM.
4. Click **Autocomplete**.

The blue piano-roll notes are input. Green notes come from the model. The app records note duration and folds sustain-pedal time into it before inference.
