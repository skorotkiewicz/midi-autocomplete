# midilm

`midilm` implements the model described in [Training a 125M-parameter Model to Autocomplete Piano](https://simedw.com/2026/08/20/midi-autocomplete/). The article does not publish source, weights, training data, or every architecture dimension, so this is a compatible implementation rather than the private iPhone model.

Each autoregressive step predicts one complete note:

```text
[event_type, pitch, delta_onset, duration, velocity]
```

It uses the published vocabularies and 24 timing steps per quarter note. The backbone is a causal decoder with RMSNorm, RoPE, self-attention, and SwiGLU blocks. Each field has its own input embedding and output head. A small nested decoder conditions later fields on earlier predictions. MIDI preprocessing bakes sustain into duration and sorts simultaneous notes by pitch.

## Set up

```sh
cd midilm
uv sync --extra cpu             # or: uv sync --extra gpu
uv run --extra cpu midilm self-test
```

## Prepare a Hugging Face dataset

For repositories containing raw `.mid`, `.midi`, or ZIP files:

```sh
uv run --extra cpu midilm prepare owner/dataset datasets/my-dataset
```

The command downloads only MIDI and ZIP files, safely extracts MIDI files from ZIP archives, and reports the number found. Run `hf auth login` first for private or gated datasets. Parquet/Arrow datasets need a small adapter for their specific column schema.

## Train

Pass the prepared directory, or any directory containing piano `.mid` or `.midi` files:

```sh
uv run --extra gpu midilm train /path/to/midi \
  --size small \
  --batch-size 8 \
  --output checkpoints/small.pt
```

The presets are approximately the article's 33M, 64M, and 125M classes. Training sums cross-entropy over all five output heads, applies transposition, tempo, duration, velocity, and dropped-note augmentation, and ramps scheduled sampling to 50%. Corpus selection, cleaning, deduplication, and train/validation splitting remain the caller's job because the original dataset is not public.

For a quick pipeline check, create an untrained tiny checkpoint. It will produce noise and is not a substitute for training:

```sh
uv run --extra cpu midilm init --size tiny --output checkpoints/tiny.pt
```

## DPO post-training

Preferences use JSON Lines. Notes contain `[pitch, delta, duration, velocity]` in quantized step values.

```json
{"prompt":[[60,0,24,80]],"chosen":[[64,24,24,80]],"rejected":[[61,192,1,124]]}
```

```sh
uv run --extra gpu midilm dpo checkpoints/small.pt preferences.jsonl \
  --beta 0.03 \
  --output checkpoints/dpo.pt
```

Point the GTK app's model field at the resulting checkpoint. The app keeps one Python inference process alive, captures real MIDI input, and sends generated notes to the selected MIDI output.
