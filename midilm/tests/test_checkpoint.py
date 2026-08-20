from unittest import TestCase

from midilm.model import parse_hf_url


class CheckpointTest(TestCase):
    def test_parses_hf_resolve_url_with_query(self) -> None:
        repo, revision, filename = parse_hf_url(
            "https://huggingface.co/Grizzlykw/midilm/resolve/main/medium.resume.pt?download=true"
        )
        self.assertEqual((repo, revision, filename), ("Grizzlykw/midilm", "main", "medium.resume.pt"))

    def test_parses_nested_resolve_url(self) -> None:
        repo, revision, filename = parse_hf_url(
            "https://huggingface.co/Grizzlykw/midilm/resolve/refs%2Fpr%2F3/checkpoints/medium.pt"
        )
        self.assertEqual((repo, revision, filename), ("Grizzlykw/midilm", "refs%2Fpr%2F3", "checkpoints/medium.pt"))

    def test_rejects_non_hf_host(self) -> None:
        with self.assertRaises(ValueError):
            parse_hf_url("https://example.com/models/medium.pt")

    def test_rejects_url_without_resolve_segment(self) -> None:
        with self.assertRaises(ValueError):
            parse_hf_url("https://huggingface.co/Grizzlykw/midilm/blob/main/medium.pt")