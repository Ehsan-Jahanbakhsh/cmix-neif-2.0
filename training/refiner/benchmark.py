"""Score byte-model predictions or run a verified full-mixer compression benchmark."""
import argparse
from pathlib import Path
import subprocess
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path, help="directory containing sample-mode binaries and weights")
    parser.add_argument("--sample", type=Path, help="input bytes from cmix's frozen alphabet")
    parser.add_argument("--model", choices=("both", "transformer", "refiner"), default="both")
    parser.add_argument("--score-only", action="store_true",
                        help="report next-byte log loss without creating a compressed payload")
    args = parser.parse_args()
    build = args.build.resolve()
    sample = (args.sample or build / "heldout.bin").resolve()
    length = sample.stat().st_size
    if not length:
        parser.error("sample is empty")
    for model, weights in (("transformer", "weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5"),
                           ("refiner", "recursive-refiner.rrb")):
        if args.model not in ("both", model):
            continue
        binary = (build / f"cmix-{model}").resolve(strict=True)
        checkpoint = (build / weights).resolve(strict=True)
        # PPMd creates a sparse scratch file in the current working directory.
        with tempfile.TemporaryDirectory(prefix=f"cmix-{model}-") as work:
            (Path(work) / weights).symlink_to(checkpoint)
            if args.score_only:
                print(model, flush=True)
                subprocess.run([binary, "--sample-score", sample], cwd=work, check=True)
                continue
            payload = Path(work) / "payload"
            start = time.monotonic()
            subprocess.run([binary, "--sample-write", sample, payload], cwd=work, check=True)
            seconds = time.monotonic() - start
            subprocess.run([binary, "--sample-verify", sample, payload], cwd=work, check=True)
            size = payload.stat().st_size
        print(f"{model}: {size} payload bytes, {8*size/length:.4f} bits/input byte, "
              f"{seconds:.2f}s encode including startup, {checkpoint.stat().st_size} weight bytes, "
              f"{length} decoded bytes verified", flush=True)


if __name__ == "__main__":
    main()
