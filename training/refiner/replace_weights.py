"""Install trained Q4 refiner weights in a compatible packaged compressor."""
import argparse
import os
from pathlib import Path
import struct
from tempfile import NamedTemporaryFile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("template", type=Path, help="comp9 built with the Q4 refiner")
    parser.add_argument("weights", type=Path, help="RRB2 or RRB3 checkpoint exported by the refiner trainer")
    parser.add_argument("output", type=Path, help="path for a new executable")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output path")
    image = args.template.read_bytes()
    weights = args.weights.read_bytes()
    if len(image) < 12 or image[:4] != b"\x7fELF":
        parser.error("expected a packaged Linux executable")
    hints, order, previous = struct.unpack_from("<3i", image, len(image)-12)
    start = len(image)-12-previous
    if (min(hints, order, previous) <= 0 or start-hints-order <= 0 or
            image[start:start+4] not in (b"RRB2", b"RRB3")):
        parser.error("template must contain a Q4 refiner checkpoint")
    if len(weights) < 32 or weights[:4] not in (b"RRB2", b"RRB3") or len(weights) >= 2**31:
        parser.error("expected an RRB2 or RRB3 checkpoint from the refiner trainer")
    if weights[:4] == b"RRB3" and image[start:start+4] != b"RRB3":
        parser.error("for score feedback, build a template with REFINER_WEIGHTS pointing to an RRB3 checkpoint")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = NamedTemporaryFile(dir=args.output.parent, delete=False)
    try:
        with pending as out:
            out.write(image[:start])
            out.write(weights)
            out.write(struct.pack("<3i", hints, order, len(weights)))
        os.chmod(pending.name, 0o755)
        os.link(pending.name, args.output)
    finally:
        Path(pending.name).unlink(missing_ok=True)
    print(f"{args.output}: {args.output.stat().st_size:,} bytes, {len(weights):,} weight bytes")


if __name__ == "__main__":
    main()
