from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from zipfile import ZipFile

from midilm.prepare import extract_zip, midi_files


class PrepareTest(TestCase):
    def test_extracts_only_midi(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "music.zip"
            with ZipFile(archive, "w") as output:
                output.writestr("piano/song.MID", b"MThd")
                output.writestr("notes.txt", b"ignored")

            extract_zip(archive, root)

            self.assertEqual([path.name for path in midi_files(root)], ["song.MID"])

    def test_rejects_path_traversal(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "unsafe.zip"
            with ZipFile(archive, "w") as output:
                output.writestr("../escape.mid", b"MThd")

            with self.assertRaisesRegex(ValueError, "unsafe archive path"):
                extract_zip(archive, root)
