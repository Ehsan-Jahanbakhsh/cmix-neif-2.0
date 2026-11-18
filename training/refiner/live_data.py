"""Prepare enwik8 or enwik9 and stream its training rows from Linux."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile

BASE = "35d1957a5074db802fc134126afc77b76d931761"
ROOT = Path(__file__).resolve().parents[2]
ENWIK9 = "159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc"
ENWIK8 = "2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
FILES = {
    "censpruef_kanon.bin": "c10b5f3d6adf513c50e2ab44a0d5383d05dcb4d54a71aa96e2366b3647d73d16",
    "censpruef_vs3.bin": "ed68bb4becbaae3ef094e920ba6dcf115b5aa7839a23502aea48c579d92412d5",
    "english.dic": "0ab9f2d2733003577b53b206791f382fd31b91eff9975041b2173bb5585c5535",
}


def verify(path, expected):
    with path.open("rb") as f:
        if hashlib.file_digest(f, "sha256").hexdigest() != expected:
            raise ValueError(f"upstream checksum mismatch for {path}")


def build_identity(binary, dataset):
    result = dict(dataset=dataset, producer=hashlib.sha256(binary.read_bytes()).hexdigest())
    if dataset == "enwik9":
        result["patch"] = hashlib.sha256(Path(__file__).with_name("live_rows.patch").read_bytes()).hexdigest()
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
        subprocess.run(["git", "-C", ROOT, "archive", "-o", archive, BASE, *paths], check=True)
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


def replay(work, dataset):
    work = work.resolve()
    if json.loads((work/"build.json").read_text()) != build_identity(work/"cmix_live", dataset):
        raise ValueError("dataset, row producer or source patch differs from the prepared build")
    for name, sha in (FILES if dataset == "enwik9" else {"enwik8": ENWIK8}).items():
        verify(work/name, sha)
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
                        shutil.copyfileobj(rows, sys.stdout.buffer, 1 << 16)
                    sys.stdout.buffer.flush()
                    if child.wait():
                        raise RuntimeError("cmix training-row producer failed")
                finally:
                    signal.signal(signal.SIGTERM, previous)
                    if child.poll() is None:
                        child.terminate()
        finally:
            os.close(read_end)
            if write_end is not None:
                os.close(write_end)


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
    args = p.parse_args()
    if args.command == "prepare":
        prepare(args)
    elif args.command == "prepare-enwik8":
        prepare_enwik8(args)
    else:
        replay(args.work, args.dataset)
