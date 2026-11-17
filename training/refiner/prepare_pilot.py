"""Prepare disjoint byte-prediction excerpts from the original enwik9 coder stream."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from prepare_demo import byte_ids, prepare_document

STREAM_SHA256 = "c35d3073eeba70f22a2e08a50202b53eb95466e3e9f1f2702f6f9b3f96fd8718"
# Leave the metadata zone (13,023,920 positions including its six-byte header) behind.
BODY_START = 13_023_920


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("binary", type=Path, help="cmix built with SAMPLE=1")
    p.add_argument("stream", type=Path, help="upstream 501,062,281-byte preprocessed stream")
    p.add_argument("out", type=Path, help="new output directory")
    p.add_argument("--excerpts", type=int, default=16, help="one in each group of four is held out")
    p.add_argument("--excerpt-bytes", type=int, default=4096)
    args = p.parse_args()
    binary = args.binary.resolve(strict=True)
    if args.excerpts < 4 or args.excerpts % 4 or args.excerpt_bytes < 2:
        p.error("use a positive multiple of four excerpts and at least two bytes per excerpt")
    if args.out.exists():
        p.error("choose a new output directory")
    body_size = args.stream.stat().st_size - BODY_START
    if args.excerpts*args.excerpt_bytes > body_size:
        p.error("excerpts do not fit in the stream body")
    lookup = byte_ids()
    with args.stream.open("rb") as src:
        digest = hashlib.file_digest(src, "sha256").hexdigest()
        if digest != STREAM_SHA256:
            p.error("expected the original enwik9 coder stream (see training/README.md section 2)")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=args.out.parent) as stage:
            out = Path(stage) / "data"
            out.mkdir()
            docs = {"train": [], "valid": []}
            for i in range(args.excerpts):
                offset = BODY_START + (2*i+1)*body_size//(2*args.excerpts) - args.excerpt_bytes//2
                src.seek(offset)
                data = src.read(args.excerpt_bytes)
                split = "valid" if i % 4 == 3 else "train"
                name = f"{split}-{i:03d}"
                prepare_document(binary, out, name, data, lookup)
                if split == "valid":
                    (out/f"{name}.bin").write_bytes(data)
                    if not docs[split]:
                        (out/"heldout.bin").write_bytes(data[:4096])
                docs[split].append({"name": name, "offset": offset, "bytes": len(data)})
            (out/"pilot.json").write_text(json.dumps(dict(sha256=digest, **docs), indent=2) + "\n")
            out.rename(args.out)
    print(f"{len(docs['train'])} training excerpts, {len(docs['valid'])} held out, "
          f"{args.excerpts*args.excerpt_bytes*208*2:,} bytes of priors")


if __name__ == "__main__":
    main()
