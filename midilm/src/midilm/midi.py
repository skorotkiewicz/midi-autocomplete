from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mido

from .model import encode_note


@dataclass
class ActiveNote:
    onset: int
    velocity: int
    released: bool = False


def read_midi(path: Path) -> list[list[int]]:
    """Read a MIDI file and bake sustain-pedal time into note durations."""
    midi = mido.MidiFile(path)
    tick = 0
    sustain = [False] * 16
    active: dict[tuple[int, int], ActiveNote] = {}
    notes: list[tuple[int, int, int, int]] = []

    def finish(key: tuple[int, int], end: int) -> None:
        note = active.pop(key, None)
        if note is not None:
            notes.append((note.onset, key[1], max(1, end - note.onset), note.velocity))

    for message in mido.merge_tracks(midi.tracks):
        tick += message.time
        if message.is_meta or getattr(message, "channel", -1) == 9:
            continue
        channel = message.channel
        if message.type == "control_change" and message.control == 64:
            was_down = sustain[channel]
            sustain[channel] = message.value >= 64
            if was_down and not sustain[channel]:
                for key, note in list(active.items()):
                    if key[0] == channel and note.released:
                        finish(key, tick)
        elif message.type == "note_on" and message.velocity > 0:
            key = (channel, message.note)
            finish(key, tick)  # Retrigger cuts a sustained note.
            active[key] = ActiveNote(tick, message.velocity)
        elif message.type in ("note_off", "note_on"):
            key = (channel, message.note)
            if key in active:
                if sustain[channel]:
                    active[key].released = True
                else:
                    finish(key, tick)

    for key in list(active):
        finish(key, tick)
    notes.sort(key=lambda note: (note[0], note[1]))

    previous_onset = 0
    encoded = []
    for onset, pitch, duration, velocity in notes:
        delta_ticks = onset - previous_onset
        delta_steps = round(delta_ticks * 24 / midi.ticks_per_beat)
        duration_steps = round(duration * 24 / midi.ticks_per_beat)
        encoded.append(encode_note(pitch, delta_steps, duration_steps, velocity))
        previous_onset = onset
    return encoded
