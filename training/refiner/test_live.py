"""Regression checks for the live row reader and training restart."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

import train_live as live


def fixture(lengths):
    rng = np.random.default_rng(22)
    docs = []
    for n in lengths:
        doc = np.zeros(n, live.ROW)
        doc["start"][0] = 1
        doc["token"] = rng.integers(0, 208, n)
        doc["target"][:-1] = doc["token"][1:]
        doc["target"][::5] = live.MASK
        doc["target"][-1] = live.MASK
        doc["prior"] = np.float16(1./208)
        docs.append(doc)
    return docs, b"RRL1" + b"".join(doc.tobytes() for doc in docs)


class LiveTrainingTest(unittest.TestCase):
    def test_wsl_replay_without_repository_mount(self):
        # Stand in for wsl.exe by running the delivered script in an empty cwd.
        popen = subprocess.Popen
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work = root/"data"
            work.mkdir()
            binary = work/"cmix_live"
            binary.write_bytes(b"test")
            digest = hashlib.sha256(binary.read_bytes()).hexdigest()
            patch_sha = hashlib.sha256(Path(live.__file__).with_name("live_rows.patch").read_bytes()).hexdigest()
            (work/"build.json").write_text(json.dumps(dict(dataset="enwik9", producer=digest, patch=patch_sha)))
            args = argparse.Namespace(wsl="Ubuntu", dataset="enwik9", work="/dataset", rows_file=None, run_dir=root)

            def launch(cmd, **kwargs):
                self.assertEqual(cmd[:8], ["wsl.exe", "-d", "Ubuntu", "--cd", "/", "--exec", "python3", "-"])
                replay = cmd[8:]
                self.assertEqual(replay[:2], ["replay", "/dataset"])
                replay[1] = str(work)
                return popen([sys.executable, "-", *replay], **dict(kwargs, cwd=root))

            with patch.object(live.subprocess, "Popen", launch):
                # Reaching the first corpus file proves the manifest and source hash passed.
                with self.assertRaisesRegex(RuntimeError, "censpruef_kanon.bin"):
                    with live.source(args):
                        self.fail("an empty producer cannot provide training rows")

    def test_boundaries_and_targets(self):
        expected, data = fixture([4096, 1, 4101, 35])
        actual = list(live.documents(io.BytesIO(data)))
        self.assertEqual(len(actual), len(expected))
        for original, doc in zip(expected, actual):
            np.testing.assert_array_equal(doc, original)
            targets = [c["target"][c["target"] != live.MASK] for c in live.windows(doc, 32, 16)]
            result = np.concatenate(targets) if targets else np.array([], dtype="u1")
            np.testing.assert_array_equal(result, doc["target"][doc["target"] != live.MASK])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            list(live.documents(io.BytesIO(data[:-1])))

    def test_resume_and_plateau(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rows = root/"fixture"
            rows.write_bytes(fixture([12]*80)[1])
            args = argparse.Namespace(rows_file=rows, dataset="enwik9", run_dir=root/"control", device="cpu", threads=1,
                    seed=1, batch=4, context=8, window=4, buffer_windows=16, eval_windows=8,
                    lr=2e-4, save_every=4, min_delta=100., patience=1, max_epochs=None)
            args.run_dir.mkdir()
            live.fit(args)
            args.run_dir = root/"resume"
            args.run_dir.mkdir()
            save = live.atomic_save
            def interrupt(state, path):
                save(state, path)
                raise InterruptedError
            with patch.object(live, "atomic_save", interrupt), self.assertRaises(InterruptedError):
                live.fit(args)
            live.fit(args)
            control = torch.load(root/"control/state.pt", map_location="cpu", weights_only=True)
            resumed = torch.load(root/"resume/state.pt", map_location="cpu", weights_only=True)
            self.assertEqual(resumed["epoch"], 2)
            self.assertEqual(resumed["history"], control["history"])
            self.assertEqual(resumed["panel_sha"], control["panel_sha"])
            self.assertEqual((root/"control/best.rrb").read_bytes(), (root/"resume/best.rrb").read_bytes())


if __name__ == "__main__":
    unittest.main()
