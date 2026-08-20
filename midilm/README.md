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

### Example repositories

Small classical piano collection, 228 raw MIDI files, CC BY 4.0:

```sh
uv run --extra cpu midilm prepare \
  xenon111/classical-piano-midi \
  datasets/classical-piano
```

Larger classical collection, 4,796 raw MIDI files, tagged MIT on its dataset card:

```sh
uv run --extra cpu midilm prepare \
  drengskapur/midi-classical-music \
  datasets/classical-music
```

Large public-domain piano collection distributed as a ZIP, 10K–100K items, CC BY-NC-SA 4.0:

```sh
uv run --extra cpu midilm prepare \
  asigalov61/Pub-Piano-MIDI-Dataset \
  datasets/pub-piano
```

Check each dataset card and source-material license before training or distributing a model. A repository license does not necessarily settle the copyright status of every included composition or performance.

## Train

Pass the prepared directory, or any directory containing piano `.mid` or `.midi` files:

```sh
uv run --extra gpu midilm train /path/to/midi \
  --size small \
  --batch-size 8 \
  --output checkpoints/small.pt
```

The presets are approximately the article's 33M, 64M, and 125M classes. Training sums cross-entropy over all five output heads, applies transposition, tempo, duration, velocity, and dropped-note augmentation, ramps scheduled sampling to 50%, warms up the learning rate linearly, then decays it with a cosine schedule to the end of training. A held-out fraction of MIDI files yields a validation loss each epoch, and the best model is written to `{output}.best.pt`. Corpus selection, cleaning, deduplication, and train/validation splitting remain the caller's job because the original dataset is not public.

Tune the schedule and validation split explicitly:

```sh
uv run --extra gpu midilm train /path/to/midi \
  --warmup-fraction 0.05 \
  --val-fraction 0.05
```

After each epoch, training writes a lean GUI checkpoint such as `medium.pt` and a `medium.resume.pt` sidecar containing optimizer and progress state. Resume with the same data and training settings:

```sh
uv run --extra gpu midilm train /path/to/midi \
  --resume checkpoints/medium.pt \
  --epochs 30 \
  --batch-size 32 \
  --workers 8 \
  --learning-rate 2e-4 \
  --scheduled-sampling 0.5 \
  --device cuda
```

`--epochs` is the total target, not the number of additional epochs. Checkpoints created before resume support was added continue from weights with a fresh optimizer and restart epoch counting.

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
