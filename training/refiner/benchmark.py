"""Score byte-model predictions or run a verified full-mixer compression benchmark."""
import argparse
import json
import math
from pathlib import Path
import re
import struct
import subprocess
import tempfile
import time


def score_prior(samples):
    """Use the refiner's prior floor with a unit gate and zero neural logits."""
    floor = math.exp(-12)
    row = struct.Struct("<208e")
    total_bits, total_rows = 0.0, 0
    for sample in samples:
        ids = sample.with_suffix(".ids").read_bytes()
        prior_path = sample.with_suffix(".f16")
        if len(ids) != sample.stat().st_size or len(ids)*row.size != prior_path.stat().st_size:
            raise ValueError(f"mismatched prior or ID trace for {sample}")
        with prior_path.open("rb") as trace:
            for target in ids[1:]:
                probs = row.unpack(trace.read(row.size))
                total_bits += math.log2(sum(max(floor, p) for p in probs) / max(floor, probs[target]))
        total_rows += len(ids) - 1
    print(f"PPMd with byte-head floor: {total_bits/total_rows:.6f} bits/scored byte, {total_rows} rows", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path, help="directory containing sample-mode binaries and weights")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--sample", type=Path, help="input bytes from cmix's frozen alphabet")
    source.add_argument("--pilot", type=Path, help="score every held-out excerpt in a prepared pilot directory")
    parser.add_argument("--ppmd-baseline", action="store_true", help="include the PPMd-only control for --pilot")
    parser.add_argument("--model", choices=("both", "transformer", "refiner"), default="both")
    parser.add_argument("--refiner-weights", type=Path, help="override the refiner checkpoint in the build directory")
    parser.add_argument("--score-only", action="store_true",
                        help="report next-byte log loss without creating a compressed payload")
    args = parser.parse_args()
    build = args.build.resolve()
    if args.pilot:
        records = json.loads((args.pilot / "pilot.json").read_text())["valid"]
        samples = [(args.pilot / (record["name"] + ".bin")).resolve() for record in records]
    else:
        samples = [(args.sample or build / "heldout.bin").resolve()]
    if not samples or any(path.stat().st_size < (2 if args.pilot else 1) for path in samples):
        parser.error("provide nonempty samples with at least two bytes per pilot excerpt")
    if args.ppmd_baseline:
        if not args.pilot:
            parser.error("--ppmd-baseline requires --pilot")
        score_prior(samples)
    for model, weights in (("transformer", "weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5"),
                           ("refiner", "recursive-refiner.rrb")):
        if args.model not in ("both", model):
            continue
        binary = (build / f"cmix-{model}").resolve(strict=True)
        checkpoint = (args.refiner_weights if model == "refiner" and args.refiner_weights else build / weights)
        checkpoint = checkpoint.resolve(strict=True)
        if args.score_only or args.pilot:
            total_bits, total_rows, elapsed = 0.0, 0, 0.0
            for sample in samples:
                # Reset both the neural predictor and the PPMd scratch file for every excerpt.
                with tempfile.TemporaryDirectory(prefix=f"cmix-{model}-") as work:
                    (Path(work) / weights).symlink_to(checkpoint)
                    start = time.monotonic()
                    run = subprocess.run([binary, "--sample-score", sample], cwd=work, check=True,
                                         stdout=subprocess.PIPE, text=True)
                    seconds = time.monotonic() - start
                match = re.search(r"byte model: ([\d.]+) bits/scored byte \((\d+) live rows", run.stdout)
                if not match:
                    raise RuntimeError(f"could not read byte-model score from {run.stdout!r}")
                loss, rows = float(match[1]), int(match[2])
                total_bits += loss * rows
                total_rows += rows
                elapsed += seconds
                print(f"{model} {sample.name}: {loss:.6f} bits/scored byte, {rows} rows, {seconds:.3f}s", flush=True)
            print(f"{model} total: {total_bits/total_rows:.6f} bits/scored byte, {total_rows} rows, "
                  f"{elapsed:.3f}s including predictor startup", flush=True)
            continue
        sample = samples[0]
        length = sample.stat().st_size
        with tempfile.TemporaryDirectory(prefix=f"cmix-{model}-") as work:
            (Path(work) / weights).symlink_to(checkpoint)
            payload = Path(work) / "payload"
            start = time.monotonic()
            subprocess.run([binary, "--sample-write", sample, payload], cwd=work, check=True)
            seconds = time.monotonic() - start
            start = time.monotonic()
            subprocess.run([binary, "--sample-verify", sample, payload], cwd=work, check=True)
            decode_seconds = time.monotonic() - start
            size = payload.stat().st_size
        print(f"{model}: {size} payload bytes, {8*size/length:.4f} bits/input byte, "
              f"{seconds:.2f}s encode, {decode_seconds:.2f}s decode and verify including startup, "
              f"{checkpoint.stat().st_size} weight bytes, "
              f"{length} decoded bytes verified", flush=True)


if __name__ == "__main__":
    main()
