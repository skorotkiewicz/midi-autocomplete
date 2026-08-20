from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

EVENTS = 5  # PAD, BOS, EOS, NOTE, MASK
PITCHES = 129  # pad + MIDI 0..127
DELTAS = tuple(range(49)) + (72, 96, 144, 192)
DURATIONS = tuple(range(1, 97)) + (144, 192, 288, 384)
VELOCITIES = tuple(range(4, 125, 8))
VOCAB_SIZES = (EVENTS, PITCHES, len(DELTAS), len(DURATIONS), len(VELOCITIES))
NOTE = 3


@dataclass(frozen=True)
class ModelConfig:
    width: int
    layers: int
    heads: int
    hidden: int
    context: int = 512

    @classmethod
    def preset(cls, name: str) -> "ModelConfig":
        return {
            "small": cls(512, 8, 8, 1792),
            "medium": cls(640, 12, 10, 1856),
            "large": cls(768, 16, 12, 2304),
            "tiny": cls(128, 2, 4, 256, 64),
        }[name]


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, context: int) -> None:
        super().__init__()
        inverse = 1.0 / (10000 ** (torch.arange(0, head_dim, 2).float() / head_dim))
        positions = torch.arange(context).float()
        angles = torch.outer(positions, inverse)
        self.register_buffer("cos", angles.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", angles.sin()[None, None, :, :], persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        length = x.size(-2)
        even, odd = x[..., ::2], x[..., 1::2]
        cos, sin = self.cos[:, :, :length], self.sin[:, :, :length]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.attention_norm = nn.RMSNorm(config.width)
        self.qkv = nn.Linear(config.width, config.width * 3, bias=False)
        self.attention_out = nn.Linear(config.width, config.width, bias=False)
        self.rope = RotaryEmbedding(self.head_dim, config.context)
        self.mlp_norm = nn.RMSNorm(config.width)
        self.gate_up = nn.Linear(config.width, config.hidden * 2, bias=False)
        self.mlp_out = nn.Linear(config.hidden, config.width, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).chunk(3, dim=-1)
        split = lambda value: value.view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        attention = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
        x = x + self.attention_out(attention.transpose(1, 2).reshape(batch, length, width))
        gate, up = self.gate_up(self.mlp_norm(x)).chunk(2, dim=-1)
        return x + self.mlp_out(F.silu(gate) * up)


class NestedStep(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.RMSNorm(width)
        self.linear = nn.Linear(width, width, bias=False)

    def forward(self, hidden: Tensor, field_embedding: Tensor) -> Tensor:
        return hidden + F.silu(self.linear(self.norm(hidden + field_embedding)))


class MidiLM(nn.Module):
    """One transformer step predicts one complete five-field note event."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = nn.ModuleList(nn.Embedding(size, config.width) for size in VOCAB_SIZES)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.norm = nn.RMSNorm(config.width)
        self.nested = nn.ModuleList(NestedStep(config.width) for _ in range(4))
        self.heads = nn.ModuleList(nn.Linear(config.width, size, bias=False) for size in VOCAB_SIZES)

    def backbone(self, notes: Tensor) -> Tensor:
        hidden = sum(embedding(notes[..., field]) for field, embedding in enumerate(self.embeddings))
        for block in self.blocks:
            hidden = block(hidden)
        return self.norm(hidden)

    def forward(self, notes: Tensor, targets: Tensor, scheduled_sampling: float = 0.0) -> tuple[Tensor, ...]:
        hidden = self.backbone(notes)
        logits: list[Tensor] = [self.heads[0](hidden)]
        for field in range(4):
            value = targets[..., field]
            if scheduled_sampling and self.training:
                predicted = logits[-1].argmax(dim=-1)
                use_prediction = torch.rand_like(value, dtype=torch.float32) < scheduled_sampling
                value = torch.where(use_prediction, predicted, value)
            hidden = self.nested[field](hidden, self.embeddings[field](value))
            logits.append(self.heads[field + 1](hidden))
        return tuple(logits)

    @torch.inference_mode()
    def next_note(self, notes: Tensor, temperature: float = 0.9, top_k: int = 16) -> Tensor:
        hidden = self.backbone(notes)[:, -1]
        values = []
        for field, head in enumerate(self.heads):
            logits = head(hidden) / max(temperature, 1e-4)
            if field == 0:
                logits[:, [0, 1, 4]] = -torch.inf
                if notes.size(1) < 4:
                    logits[:, 2] = -torch.inf
            k = min(top_k, logits.size(-1))
            cutoff = logits.topk(k).values[:, -1, None]
            logits = logits.masked_fill(logits < cutoff, -torch.inf)
            value = torch.multinomial(logits.softmax(dim=-1), 1).squeeze(-1)
            values.append(value)
            if field < 4:
                hidden = self.nested[field](hidden, self.embeddings[field](value))
        return torch.stack(values, dim=-1)

    @torch.inference_mode()
    def generate(self, prompt: Tensor, count: int, temperature: float = 0.9, top_k: int = 16) -> Tensor:
        notes = prompt
        generated = []
        for _ in range(count):
            context = notes[:, -self.config.context :]
            note = self.next_note(context, temperature, top_k)
            if note[0, 0].item() == 2:
                break
            if note[0, 0].item() != NOTE:
                note[:, 0] = NOTE
            generated.append(note)
            notes = torch.cat((notes, note[:, None]), dim=1)
            if notes.size(1) >= self.config.context:
                notes = notes[:, -384:]
        return torch.stack(generated, dim=1) if generated else notes[:, :0]


def atomic_save(path: Path, checkpoint: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_checkpoint(path: Path, model: MidiLM) -> None:
    atomic_save(path, {"config": asdict(model.config), "model": model.state_dict()})


def resume_checkpoint_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.resume{path.suffix}")


def save_training_checkpoint(
    path: Path,
    model: MidiLM,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    atomic_save(
        resume_checkpoint_path(path),
        {
            "config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "step": step,
        },
    )


def load_training_checkpoint(path: Path, device: str) -> dict:
    sidecar = resume_checkpoint_path(path)
    return torch.load(sidecar if sidecar.exists() else path, map_location=device, weights_only=True)


def load_checkpoint(path: Path, device: str = "cpu") -> MidiLM:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = MidiLM(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval()


def nearest_id(value: int, vocabulary: tuple[int, ...]) -> int:
    return min(range(len(vocabulary)), key=lambda index: abs(vocabulary[index] - value))


def encode_note(pitch: int, delta: int, duration: int, velocity: int) -> list[int]:
    return [NOTE, max(0, min(127, pitch)) + 1, nearest_id(delta, DELTAS), nearest_id(duration, DURATIONS), nearest_id(velocity, VELOCITIES)]


def decode_note(note: Tensor) -> tuple[int, int, int, int]:
    values = note.tolist()
    return (
        max(0, values[1] - 1),
        DELTAS[values[2]],
        DURATIONS[values[3]],
        VELOCITIES[values[4]],
    )
