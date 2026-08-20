from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import mido

from midilm.midi import read_midi


class MidiTest(TestCase):
    def test_ignores_channel_less_sysex_events(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "sysex.mid"
            midi = mido.MidiFile(ticks_per_beat=480)
            track = mido.MidiTrack(
                [
                    mido.Message("sysex", data=(65, 16, 66), time=0),
                    mido.Message("note_on", note=60, velocity=80, time=0),
                    mido.Message("note_off", note=60, velocity=0, time=480),
                ]
            )
            midi.tracks.append(track)
            midi.save(path)

            notes = read_midi(path)

            self.assertEqual(len(notes), 1)
