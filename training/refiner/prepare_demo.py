"""Prepare the profiling split and collect each document's causal cmix priors."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]


def byte_ids():
    bitmap = (ROOT / "training/tf/2_package/vocab-bitmap.u8").read_bytes()
    alphabet = (i for i in range(256) if bitmap[i // 8] & (1 << (i % 8)))
    return {b: i for i, b in enumerate(alphabet)}


def prepare_document(binary, out, name, data, lookup):
    ids = bytes(lookup[b] for b in data)
    with tempfile.TemporaryDirectory() as tmp:
        input_path = Path(tmp) / "body"
        input_path.write_bytes(data)
        prior_path = Path(tmp) / "prior"
        subprocess.run([binary, "--sample-priors", input_path, prior_path], cwd=tmp, check=True)
        if prior_path.stat().st_size != len(data)*208*2:
            raise ValueError("incomplete PPMd trace")
        shutil.copyfile(prior_path, out / f"{name}.f16")
    (out / f"{name}.ids").write_bytes(ids)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path, help="cmix built with SAMPLE=1")
    parser.add_argument("out", type=Path)
    parser.add_argument("--input", type=Path, default=ROOT / "prof_input/input",
                        help="small raw text corpus, default: upstream profiling text")
    args = parser.parse_args()
    binary = args.binary.resolve(strict=True)
    lookup = byte_ids()
    source = args.input.read_bytes()
    raw = bytes(b for b in source.lower() if b in lookup)
    split = len(raw)*9//10
    if min(split, len(raw)-split) < 2:
        parser.error("both splits need at least two bytes after lowercasing and alphabet filtering")
    print(f"{len(source)} input bytes, {len(source)-len(raw)} outside alphabet, "
          f"{split} training bytes, {len(raw)-split} validation bytes", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, data in (("train", raw[:split]), ("valid", raw[split:])):
        prepare_document(binary, args.out, name, data, lookup)
    (args.out / "heldout.bin").write_bytes(raw[split:split+4096])


if __name__ == "__main__":
    main()
