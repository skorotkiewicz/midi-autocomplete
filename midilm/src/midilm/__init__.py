from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from mido.midifiles.meta import KeySignatureError
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .midi import read_midi
from .prepare import prepare_dataset
from .model import (
    DELTAS,
    DURATIONS,
    VELOCITIES,
    MidiLM,
    ModelConfig,
    decode_note,
    encode_note,
    load_checkpoint,
    load_training_checkpoint,
    nearest_id,
    resume_checkpoint_path,
    save_checkpoint,
    save_training_checkpoint,
)

BOS = [1, 0, 0, 0, 0]
EOS = [2, 0, 0, 0, 0]
PAD = [0, 0, 0, 0, 0]


def augment(notes: list[list[int]]) -> list[list[int]]:
    if not notes:
        return notes
    pitches = [note[1] - 1 for note in notes]
    transpose = random.randint(max(-6, -min(pitches)), min(6, 127 - max(pitches)))
    tempo = random.uniform(0.9, 1.1)
    onset = 0
    performed = []
    for note in notes:
        onset += DELTAS[note[2]]
        performed.append((note[1] - 1 + transpose, onset, DURATIONS[note[3]], VELOCITIES[note[4]]))
    kept = [note for note in performed if random.random() >= 0.02] or performed[:1]
    previous_onset = None
    augmented = []
    for pitch, onset, duration, velocity in kept:
        delta = 0 if previous_onset is None else round((onset - previous_onset) * tempo)
        augmented.append(
            [
                3,
                pitch + 1,
                nearest_id(delta, DELTAS),
                nearest_id(round(duration * tempo * random.uniform(0.9, 1.1)), DURATIONS),
                nearest_id(velocity + random.randint(-4, 4), VELOCITIES),
            ]
        )
        previous_onset = onset
    return augmented


class MidiDataset(Dataset[Tensor]):
    def __init__(self, root: Path, context: int) -> None:
        self.paths = sorted(
            path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in (".mid", ".midi")
        )
        if not self.paths:
            raise ValueError(f"no .mid or .midi files under {root}")
        self.context = context

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tensor:
        for offset in range(len(self.paths)):
            try:
                notes = read_midi(self.paths[(index + offset) % len(self.paths)])
            except (EOFError, IndexError, OSError, ValueError, KeySignatureError):
                continue
            notes = augment(notes)
            break
        else:
            raise RuntimeError("dataset contains no readable MIDI files")
        if len(notes) >= self.context - 1:
            start = random.randrange(len(notes) - self.context + 2)
            sequence = notes[start : start + self.context - 1]
            sequence = ([BOS] if start == 0 else []) + sequence
        else:
            sequence = [BOS, *notes, EOS]
        sequence = sequence[: self.context + 1]
        sequence += [PAD] * (self.context + 1 - len(sequence))
        return torch.tensor(sequence, dtype=torch.long)


def masked_log_probability(model: MidiLM, sequence: Tensor) -> Tensor:
    inputs, targets = sequence[:, :-1], sequence[:, 1:]
    logits = model(inputs, targets)
    mask = targets[..., 0] != 0
    return sum(
        (F.log_softmax(field_logits, -1).gather(-1, targets[..., field, None]).squeeze(-1) * mask).sum(-1)
        for field, field_logits in enumerate(logits)
    )


def train(args: argparse.Namespace) -> None:
    device = args.device
    output = args.output or args.resume or Path(f"checkpoints/{args.size}.pt")
    checkpoint = load_training_checkpoint(args.resume, device) if args.resume else None
    config = ModelConfig(**checkpoint["config"]) if checkpoint else ModelConfig.preset(args.size)
    model = MidiLM(config).to(device)
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    dataset = MidiDataset(args.data, config.context)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)
    start_epoch = int(checkpoint.get("epoch", 0)) if checkpoint else 0
    step = int(checkpoint.get("step", 0)) if checkpoint else 0
    if checkpoint and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        print(f"resuming from epoch {start_epoch}, step {step}")
    elif checkpoint:
        print("warning: resume checkpoint has no optimizer state; continuing from weights with a fresh optimizer")
    if start_epoch >= args.epochs:
        raise ValueError(f"checkpoint already completed {start_epoch} epochs; --epochs must be larger")
    total_steps = max(1, args.epochs * len(loader))

    model.train()
    for epoch in range(start_epoch, args.epochs):
        progress = tqdm(loader, desc=f"epoch {epoch + 1}/{args.epochs}")
        for sequence in progress:
            sequence = sequence.to(device)
            inputs, targets = sequence[:, :-1], sequence[:, 1:]
            scheduled = args.scheduled_sampling * step / total_steps
            logits = model(inputs, targets, scheduled)
            mask = targets[..., 0] != 0
            losses = []
            for field, field_logits in enumerate(logits):
                per_note = F.cross_entropy(
                    field_logits.flatten(0, 1), targets[..., field].flatten(), reduction="none"
                ).view_as(mask)
                losses.append((per_note * mask).sum() / mask.sum().clamp_min(1))
            loss = sum(losses)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            step += 1
            progress.set_postfix(loss=f"{loss.item():.3f}", scheduled=f"{scheduled:.2f}")
        save_checkpoint(output, model)
        save_training_checkpoint(output, model, optimizer, epoch + 1, step)


def preference_sequence(notes: list[list[int]], context: int) -> Tensor:
    encoded = [encode_note(*note) for note in notes]
    sequence = [BOS, *encoded, EOS][: context + 1]
    sequence += [PAD] * (context + 1 - len(sequence))
    return torch.tensor(sequence, dtype=torch.long).unsqueeze(0)


def dpo(args: argparse.Namespace) -> None:
    policy = load_checkpoint(args.checkpoint, args.device).train()
    reference = load_checkpoint(args.checkpoint, args.device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.learning_rate)
    records = [json.loads(line) for line in args.preferences.read_text().splitlines() if line.strip()]
    for epoch in range(args.epochs):
        random.shuffle(records)
        progress = tqdm(records, desc=f"DPO {epoch + 1}/{args.epochs}")
        for record in progress:
            prompt = record["prompt"]
            chosen = preference_sequence(prompt + record["chosen"], policy.config.context).to(args.device)
            rejected = preference_sequence(prompt + record["rejected"], policy.config.context).to(args.device)
            policy_margin = masked_log_probability(policy, chosen) - masked_log_probability(policy, rejected)
            with torch.no_grad():
                reference_margin = masked_log_probability(reference, chosen) - masked_log_probability(reference, rejected)
            loss = -F.logsigmoid(args.beta * (policy_margin - reference_margin)).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            progress.set_postfix(loss=f"{loss.item():.3f}")
        save_checkpoint(args.output, policy)


def parse_notes(text: str) -> list[list[int]]:
    if not text:
        return []
    return [[int(value) for value in note.split(",")] for note in text.split(";")]


def run_generation(model: MidiLM, text: str, count: int, temperature: float, top_k: int) -> str:
    prompt = [BOS, *(encode_note(*note) for note in parse_notes(text))]
    tensor = torch.tensor(prompt, dtype=torch.long, device=next(model.parameters()).device).unsqueeze(0)
    generated = model.generate(tensor, count, temperature, top_k)[0]
    return ";".join(",".join(map(str, decode_note(note))) for note in generated)


def serve(args: argparse.Namespace) -> None:
    model = load_checkpoint(args.checkpoint, args.device)
    print(f"ready\t{sum(parameter.numel() for parameter in model.parameters())}", flush=True)
    for line in sys.stdin:
        try:
            command, count, temperature, top_k, notes = line.rstrip("\n").split("\t", 4)
            if command != "generate":
                raise ValueError("unknown command")
            result = run_generation(model, notes, int(count), float(temperature), int(top_k))
            print(f"notes\t{result}", flush=True)
        except Exception as error:
            print(f"error\t{error}", flush=True)


def self_test() -> None:
    config = ModelConfig.preset("tiny")
    model = MidiLM(config)
    prompt = torch.tensor([[BOS, encode_note(60, 0, 24, 80)]], dtype=torch.long).reshape(1, 2, 5)
    output = model.generate(prompt, 2)
    assert output.shape == (1, 2, 5)
    assert torch.all(output[..., 0] == 3)
    random.seed(0)
    augmented = augment([encode_note(60, 0, 24, 80), encode_note(64, 24, 24, 80)])
    assert augmented and all(note[0] == 3 for note in augmented)
    path = Path("/tmp/midilm-self-test.pt")
    optimizer = torch.optim.AdamW(model.parameters())
    targets = prompt[:, 1:]
    loss = sum(logits.mean() for logits in model(prompt[:, :-1], targets))
    loss.backward()
    optimizer.step()
    save_checkpoint(path, model)
    save_training_checkpoint(path, model, optimizer, epoch=3, step=7)
    resumed = load_training_checkpoint(path, "cpu")
    assert resumed["epoch"] == 3 and resumed["step"] == 7
    assert resumed["optimizer"]["state"]
    assert resume_checkpoint_path(path).exists()
    assert load_checkpoint(path).config == config
    print("midilm self-test passed")


def main() -> None:
    parser = argparse.ArgumentParser(prog="midilm")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("prepare", help="download MIDI files from a Hugging Face dataset")
    download.add_argument("repository", help="dataset repository, for example owner/name")
    download.add_argument("output", type=Path)
    download.add_argument("--revision")

    init = commands.add_parser("init", help="create an untrained checkpoint")
    init.add_argument("--size", choices=("tiny", "small", "medium", "large"), default="small")
    init.add_argument("--output", type=Path, default=Path("checkpoints/small.pt"))

    fit = commands.add_parser("train", help="train on a directory of MIDI files")
    fit.add_argument("data", type=Path)
    fit.add_argument("--size", choices=("small", "medium", "large"), default="small")
    fit.add_argument("--output", type=Path)
    fit.add_argument("--resume", type=Path, help="resume model, optimizer, epoch, and step from a checkpoint")
    fit.add_argument("--epochs", type=int, default=10, help="total target epochs, including completed epochs")
    fit.add_argument("--batch-size", type=int, default=4)
    fit.add_argument("--workers", type=int, default=0)
    fit.add_argument("--learning-rate", type=float, default=3e-4)
    fit.add_argument("--scheduled-sampling", type=float, default=0.5)
    fit.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    preference = commands.add_parser("dpo", help="post-train on preference JSONL")
    preference.add_argument("checkpoint", type=Path)
    preference.add_argument("preferences", type=Path)
    preference.add_argument("--output", type=Path, default=Path("checkpoints/dpo.pt"))
    preference.add_argument("--epochs", type=int, default=1)
    preference.add_argument("--beta", type=float, default=0.03)
    preference.add_argument("--learning-rate", type=float, default=1e-6)
    preference.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    server = commands.add_parser("serve", help="serve line-based inference for the desktop app")
    server.add_argument("checkpoint", type=Path)
    server.add_argument("--device", default="cpu")

    commands.add_parser("self-test")
    args = parser.parse_args()
    if args.command == "prepare":
        count = prepare_dataset(args.repository, args.output, args.revision)
        print(f"prepared {count:,} MIDI files under {args.output}")
    elif args.command == "init":
        model = MidiLM(ModelConfig.preset(args.size))
        save_checkpoint(args.output, model)
        print(f"saved {sum(parameter.numel() for parameter in model.parameters()):,} parameters to {args.output}")
    elif args.command == "train":
        train(args)
    elif args.command == "dpo":
        dpo(args)
    elif args.command == "serve":
        serve(args)
    else:
        self_test()
