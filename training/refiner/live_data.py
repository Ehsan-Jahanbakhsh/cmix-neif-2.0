"""Prepare enwik8 or enwik9 and stream its training rows from Linux."""
import argparse
from collections import deque
from contextlib import contextmanager
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import zlib

BASE = "35d1957a5074db802fc134126afc77b76d931761"
ENWIK9 = "159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc"
ENWIK8 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
ROW_BYTES = 420
CHUNK_ROWS = 32768
TOTAL_ROWS = {"enwik8": 100_000_000, "enwik9": 543_832_785}
FILES = {
    "censpruef_kanon.bin": "c10b5f3d6adf513c50e2ab44a0d5383d05dcb4d54a71aa96e2366b3647d73d16",
    "censpruef_vs3.bin": "ed68bb4becbaae3ef094e920ba6dcf115b5aa7839a23502aea48c579d92412d5",
    "english.dic": "0ab9f2d2733003577b53b206791f382fd31b91eff9975041b2173bb5585c5535",
}


def verify(path, expected):
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected:
            raise ValueError(f"upstream checksum mismatch for {path}")


def build_identity(binary, dataset, patch_sha256=None):
    result = dict(dataset=dataset, producer=hashlib.sha256(binary.read_bytes()).hexdigest())
    if dataset == "enwik9":
        if patch_sha256 is None:
            patch_sha256 = hashlib.sha256(Path(__file__).with_name("live_rows.patch").read_bytes()).hexdigest()
        result["patch"] = patch_sha256
    return result


def prepare(args):
    out = args.work.resolve()
    if out.exists():
        raise FileExistsError(f"choose a new directory for preparation: {out}")
    verify(args.enwik9, ENWIK9)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as temp:
        stage = Path(temp)
        src = stage / "source"
        src.mkdir()
        paths = ["src", "makefile", "encoder_split.lds", "runtime_exports.map",
                 "training/tf/1_dump", "dictionary/english.dic"]
        archive = stage / "source.tar"
        root = Path(__file__).resolve().parents[2]
        subprocess.run(["git", "-C", root, "archive", "-o", archive, BASE, *paths], check=True)
        subprocess.run(["tar", "xf", archive, "-C", src], check=True)
        build = src / "training/tf/1_dump"
        tools = stage / "tools"
        dest = stage / "ready"
        dest.mkdir()
        shutil.copyfile(src / "dictionary/english.dic", dest / "english.dic")
        bed = args.reuse_bed
        if bed is None:
            subprocess.run(["bash", build/"build_dump_tools.sh", src, tools, "bed"], check=True)
            bed = stage / "bed"
            bed.mkdir()
            shutil.copyfile(dest/"english.dic", bed/".dict")
            shutil.copyfile(src/"src/readalike_prepr/data/new_article_order", bed/".new_article_order")
            subprocess.run([tools/"cmix_bed", "-e", args.enwik9.resolve(), "out"], cwd=bed, check=True)
        for name in ("censpruef_kanon.bin", "censpruef_vs3.bin"):
            shutil.copyfile(bed/name, dest/name)
        for name, sha in FILES.items():
            verify(dest/name, sha)
        for patch in (build/"trainmodes.patch", Path(__file__).with_name("live_rows.patch")):
            with patch.open("rb") as f:
                subprocess.run(["patch", "-p1", "--batch"], stdin=f, cwd=src, check=True)
        with (stage/"compile.log").open("w") as log:
            result = subprocess.run(["make", "-j4", "cmix", "TRAIN_MODE=3", "CC=clang++-18 -DTRAIN_MODE=3"],
                                    cwd=src, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError((stage/"compile.log").read_text()[-6000:])
        shutil.copy2(src/"cmix", dest/"cmix_live")
        (dest/"build.json").write_text(json.dumps(build_identity(dest/"cmix_live", "enwik9")) + "\n")
        dest.rename(out)
    print(f"Prepared {out}, {sum(p.stat().st_size for p in out.iterdir()):,} bytes")


def prepare_enwik8(args):
    out = args.work.resolve()
    if out.exists():
        raise FileExistsError(f"choose a new directory for preparation: {out}")
    if args.input.stat().st_size not in (100_000_000, 1_000_000_000):
        raise ValueError("expected enwik8 or the uncompressed original enwik9")
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        dest = Path(tmp)/"ready"
        dest.mkdir()
        with args.input.open("rb") as src, (dest/"enwik8").open("wb") as sink:
            remaining = 100_000_000
            while remaining:
                piece = src.read(min(1 << 20, remaining))
                if not piece:
                    raise EOFError("truncated XML")
                sink.write(piece)
                remaining -= len(piece)
        verify(dest/"enwik8", ENWIK8)
        shutil.copy2(args.binary, dest/"cmix_live")
        (dest/"build.json").write_text(json.dumps(build_identity(dest/"cmix_live", "enwik8")) + "\n")
        dest.rename(out)
    print(f"Prepared verified enwik8 in {out}")


@contextmanager
def native_rows(work, dataset, *, stop_early=False):
    # A new scratch directory isolates each PPMd page file. Only rows go to stdout.
    with tempfile.TemporaryDirectory(prefix="refiner-ppmd-", dir=work.parent) as temp:
        run = Path(temp)
        # WSL can expose its Windows stdout as a socket that cannot be reopened
        # with fopen("/dev/fd/N"). Give the producer a regular Linux pipe.
        read_end, write_end = os.pipe()
        env = dict(os.environ, CMIX_REFINER_ROWS=f"/dev/fd/{write_end}", NEIF_TF1_WEIGHTS="none", OMP_NUM_THREADS="1")
        if dataset == "enwik9":
            (run/".dict").symlink_to(work/"english.dic")
            (run/".vs3").symlink_to(work/"censpruef_vs3.bin")
            (run/"discard").symlink_to("/dev/null")
            cmd = [work/"cmix_live", "-n", work/"censpruef_kanon.bin", "discard"]
        else:
            cmd = [work/"cmix_live", "--enwik8-rows", work/"enwik8", env["CMIX_REFINER_ROWS"]]
        for key in tuple(env):
            if key.startswith("NEIF_") and key != "NEIF_TF1_WEIGHTS":
                del env[key]
        try:
            with subprocess.Popen(cmd, cwd=run, env=env, stdout=sys.stderr, pass_fds=(write_end,)) as child:
                os.close(write_end)
                write_end = None
                def cancel(_signum, _frame):
                    raise KeyboardInterrupt
                previous = signal.signal(signal.SIGTERM, cancel)
                try:
                    with os.fdopen(read_end, "rb", closefd=False) as rows:
                        yield rows
                    if not stop_early and child.wait():
                        raise RuntimeError("cmix training-row producer failed")
                finally:
                    signal.signal(signal.SIGTERM, previous)
                    if child.poll() is None:
                        child.terminate()
        finally:
            os.close(read_end)
            if write_end is not None:
                os.close(write_end)


def build_cache(cache, identity, total, budget, open_rows, output):
    # Keep the newest consecutive chunks. A suffix can be replayed without
    # reconstructing PPMd state after the cached section.
    chunks = deque()
    used = rows = 0
    with open_rows() as stream:
        if stream.read(4) != b"RRL1":
            raise ValueError("invalid row header")
        output.write(b"RRL1")
        while raw := stream.read(CHUNK_ROWS * ROW_BYTES):
            if len(raw) != min(CHUNK_ROWS, total - rows) * ROW_BYTES:
                raise ValueError("unexpected native row count")
            output.write(raw)
            packed = zlib.compress(raw, 1)
            while chunks and used + len(packed) > budget:
                old_start, size, _ = chunks.popleft()
                (cache/f"{old_start}.zlib").unlink()
                used -= size
            if len(packed) > budget:
                raise ValueError("cache budget is smaller than one chunk")
            (cache/f"{rows}.zlib").write_bytes(packed)
            chunks.append((rows, len(packed), hashlib.sha256(packed).hexdigest()))
            used += len(packed)
            rows += len(raw) // ROW_BYTES
    if rows != total:
        raise ValueError(f"truncated native stream: {rows} of {total} rows")
    index = dict(version=1, identity=identity, total=total, chunks=list(chunks))
    (cache/"index.tmp").write_text(json.dumps(index))
    (cache/"index.tmp").replace(cache/"index.json")
    print(f"Cached {total-chunks[0][0]:,} of {total:,} rows in {used:,} bytes.", file=sys.stderr)


def replay_cache(cache, identity, total, budget, open_rows, output):
    index = json.loads((cache/"index.json").read_text())
    if (index["version"], index["identity"], index["total"]) != (1, identity, total):
        raise ValueError(f"cache belongs to different data or producer: {cache}")
    chunks = index["chunks"]
    start = chunks[0][0]
    if (not 0 <= start < total or start % CHUNK_ROWS or
            [c[0] for c in chunks] != list(range(start, total, CHUNK_ROWS))):
        raise ValueError("incomplete PPMd cache index")
    if sum(c[1] for c in chunks) > budget:
        raise ValueError("existing PPMd cache exceeds requested budget; use its original budget")
    if start:
        with open_rows(stop_early=True) as stream:
            if stream.read(4) != b"RRL1":
                raise ValueError("invalid row header")
            output.write(b"RRL1")
            remaining = start * ROW_BYTES
            while remaining:
                raw = stream.read(min(1 << 20, remaining))
                if not raw:
                    raise EOFError("native stream ended before cached suffix")
                output.write(raw)
                remaining -= len(raw)
    else:
        output.write(b"RRL1")
    print(f"Reading {total-start:,} cached rows starting at {start:,}.", file=sys.stderr)
    for offset, size, digest in chunks:
        packed = (cache/f"{offset}.zlib").read_bytes()
        if len(packed) != size or hashlib.sha256(packed).hexdigest() != digest:
            raise ValueError(f"damaged PPMd cache chunk at row {offset}")
        raw = zlib.decompress(packed)
        if len(raw) != min(CHUNK_ROWS, total-offset) * ROW_BYTES:
            raise ValueError(f"wrong row count in cached chunk at {offset}")
        output.write(raw)


def replay(work, dataset, patch_sha256=None, cache_gib=0):
    work = work.resolve()
    identity = build_identity(work/"cmix_live", dataset, patch_sha256)
    if json.loads((work/"build.json").read_text()) != identity:
        raise ValueError("dataset, row producer or source patch differs from the prepared build")
    for name, sha in (FILES if dataset == "enwik9" else {"enwik8": ENWIK8}).items():
        verify(work/name, sha)
    open_rows = partial(native_rows, work, dataset)
    output = sys.stdout.buffer
    if cache_gib:
        import fcntl
        # Leave space inside the cap for directory entries, block rounding and index.
        budget = cache_gib * 2**30 * 99 // 100
        cache = work/"refiner-row-cache-v1"
        cache.mkdir(exist_ok=True)
        with (cache/"builder.lock").open("a+b") as lock:
            if not (cache/"index.json").exists():
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    print("Another run is building the cache; streaming live for this pass.", file=sys.stderr)
                    with open_rows() as stream:
                        shutil.copyfileobj(stream, output, 1 << 16)
                    output.flush()
                    return
                if not (cache/"index.json").exists():
                    # An interrupted first pass has no published index. Rebuild once
                    # under the lock, removing only our incomplete chunk files.
                    for path in cache.glob("*.zlib"):
                        if path.stem.isdecimal():
                            path.unlink()
                    print(f"Building a reusable PPMd cache in {cache} (limit {cache_gib} GiB).", file=sys.stderr)
                    build_cache(cache, identity, TOTAL_ROWS[dataset], budget, open_rows, output)
                    output.flush()
                    return
        replay_cache(cache, identity, TOTAL_ROWS[dataset], budget, open_rows, output)
    else:
        with open_rows() as stream:
            shutil.copyfileobj(stream, output, 1 << 16)
    output.flush()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("enwik9", type=Path)
    prep.add_argument("work", type=Path)
    prep.add_argument("--reuse-bed", type=Path, help="existing upstream cmix_bed outputs")
    small = commands.add_parser("prepare-enwik8")
    small.add_argument("input", type=Path, help="uncompressed enwik8 or enwik9")
    small.add_argument("work", type=Path)
    small.add_argument("--binary", type=Path, required=True, help="fork sample binary built with SAMPLE=1")
    run = commands.add_parser("replay")
    run.add_argument("work", type=Path)
    run.add_argument("--dataset", choices=("enwik8", "enwik9"), default="enwik9")
    run.add_argument("--patch-sha256", help="enwik9 source patch digest supplied by the Windows trainer")
    run.add_argument("--cache-gib", type=int, default=0)
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "prepare-enwik8":
        prepare_enwik8(args)
    else:
        if args.cache_gib < 0:
            p.error("--cache-gib must be nonnegative")
        replay(args.work, args.dataset, args.patch_sha256, args.cache_gib)
