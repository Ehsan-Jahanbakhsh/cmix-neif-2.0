"""Train a quantization-aware refiner on live or cached enwik8 or enwik9 rows."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
from tempfile import NamedTemporaryFile
import time

from filelock import FileLock
import numpy as np
import torch
import torch.nn.functional as F

from train import PRESETS, ROOT, enable_q4, export, logits, make_model

ROW = np.dtype([("token", "u1"), ("target", "u1"), ("start", "u1"), ("reserved", "u1"),
                ("prior", "<f2", (208,))])
MASK = 255


@contextmanager
def source(args):
    if args.rows_file:
        with args.rows_file.open("rb") as f:
            yield f
        return
    producer = Path(__file__).with_name("live_data.py").resolve()
    replay = ["replay", args.work, "--dataset", args.dataset]
    if args.cache_gib:
        replay.extend(["--cache-gib", str(args.cache_gib)])
    if args.wsl:
        # The Windows repo drive can disappear from WSL between epochs. Feed the
        # script over stdin and use only Linux paths for an absolute work directory.
        cwd = "/" if PurePosixPath(args.work).is_absolute() else str(ROOT)
        cmd = ["wsl.exe", "-d", args.wsl, "--cd", cwd, "--exec", "python3", "-", *replay]
        if args.dataset == "enwik9":
            patch = producer.with_name("live_rows.patch")
            cmd.extend(["--patch-sha256", hashlib.sha256(patch.read_bytes()).hexdigest()])
    else:
        cmd = ["python3", str(producer), *replay]
    with producer.open("rb") as script, (args.run_dir/"producer.log").open("a+b") as log:
        log_start = log.tell()
        child = subprocess.Popen(cmd, cwd=ROOT, stdin=script if args.wsl else subprocess.DEVNULL,
                                 stdout=subprocess.PIPE, stderr=log)
        try:
            has_output = bool(child.stdout.peek(1))
            if has_output:
                yield child.stdout
            code = child.wait()
            if code or not has_output:
                end = log.seek(0, os.SEEK_END)
                log.seek(max(log_start, end - 2048))
                detail = log.read().decode(errors="replace").strip() or "empty row stream"
                raise RuntimeError(f"row producer exited with status {code}: {detail} (see {log.name})")
        finally:
            child.stdout.close()
            if child.poll() is None:
                child.terminate()
            child.wait()


def documents(stream):
    if stream.read(4) != b"RRL1":
        raise ValueError("missing live-row header (see producer.log)")
    pieces, length = [], 0
    while data := stream.read(4096 * ROW.itemsize):
        if len(data) % ROW.itemsize:
            raise ValueError("incomplete live row")
        rows = np.frombuffer(data, ROW)
        target = rows["target"]
        if (np.any(rows["token"] >= 216) or np.any((target >= 208) & (target != MASK)) or
                np.any(rows["start"] > 1) or np.any(rows["reserved"])):
            raise ValueError("invalid live row")
        starts = np.flatnonzero(rows["start"])
        if not pieces and (not len(starts) or starts[0] != 0):
            raise ValueError("expected a document start")
        offset = 0
        for end in starts:
            if end > offset:
                pieces.append(rows[offset:end])
                length += end - offset
            if length:
                yield np.concatenate(pieces)
            pieces, length, offset = [], 0, end
        pieces.append(rows[offset:])
        length += len(rows) - offset
        if length > 262144:
            raise ValueError("document exceeds the maximum live buffer")
    if length:
        yield np.concatenate(pieces)


def windows(doc, context, window):
    # The same loss mask and left-context convention as the cropped trainer.
    warmup = min(window, context//2)
    stride = context - warmup
    for start in range(0, len(doc), stride):
        left = max(0, start-warmup)
        part = doc[left:start+stride]
        skip = start-left
        if np.all(part["target"][skip:] == MASK):
            continue
        crop = np.zeros(context, ROW)
        crop["target"] = MASK
        crop[:len(part)] = part
        crop["target"][:skip] = MASK
        yield crop


def held_out(doc_id):
    # A stable split by entire document throughout the ordered corpus.
    value = hashlib.blake2s(doc_id.to_bytes(8, "little"), digest_size=4).digest()
    return int.from_bytes(value, "little") % 20 == 0


def batch_loss(model, batch, device):
    x = torch.from_numpy(batch["token"].astype(np.int64)).to(device)
    y_np = batch["target"].astype(np.int64)
    scored = int(np.count_nonzero(y_np != MASK))
    y_np[y_np == MASK] = -100
    y = torch.from_numpy(y_np).to(device)
    p = torch.from_numpy(batch["prior"].copy()).to(device)
    return F.cross_entropy(logits(model, x, p).flatten(0, 1), y.flatten(), reduction="sum"), scored


def train_buffer(model, optimizer, rows, rng, args):
    order = rng.permutation(len(rows))
    for start in range(0, len(rows), args.batch):
        batch = rows[order[start:start+args.batch]]
        optimizer.zero_grad(set_to_none=True)
        loss, count = batch_loss(model, batch, args.device)
        (loss/count).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
        optimizer.step()


@torch.no_grad()
def validation(model, panel, args):
    if not len(panel):
        raise ValueError("no held-out targets in this stream")
    model.eval()
    bits, count = 0., 0
    for i in range(0, len(panel), args.batch):
        loss, n = batch_loss(model, panel[i:i+args.batch], args.device)
        bits += loss.item()/math.log(2)
        count += n
    model.train()
    if not math.isfinite(bits):
        raise ValueError("non-finite validation loss")
    return bits/count, count


def atomic_save(state, path):
    with NamedTemporaryFile(dir=path.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(state, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def fit(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    dims = PRESETS[args.preset]
    model = make_model(dims, args.window, args.context, args.score_feedback).to(args.device)
    enable_q4(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    version = hashlib.sha256(Path(__file__).with_name("live_rows.patch").read_bytes()).hexdigest()
    identity = (args.dataset, version)
    if args.rows_file:
        with args.rows_file.open("rb") as f:
            identity = hashlib.file_digest(f, "sha256").hexdigest()
    settings = (identity, dims, args.context, args.window, args.batch, args.buffer_windows, args.seed,
                args.eval_windows, args.lr, args.min_delta, args.patience)
    if args.score_feedback:
        settings += ("score-feedback-v1",)
    path = args.run_dir/"state.pt"
    state = dict(settings=settings, epoch=0, buffers=0, best=math.inf, anchor=math.inf, stale=0, history=[])
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state["settings"] != settings:
            raise ValueError("data or training settings differ from the saved run")
        model.load_state_dict(state.pop("model"))
        optimizer.load_state_dict(state.pop("optimizer"))

    def save():
        atomic_save(dict(state, model=model.state_dict(), optimizer=optimizer.state_dict()), path)

    buffers = np.empty((2 if args.prefetch else 1, args.buffer_windows, args.context), ROW)
    panel = np.empty((args.eval_windows, args.context), ROW)
    limit = args.epochs if args.epochs is not None else args.max_epochs
    while limit is None or state["epoch"] < limit:
        if args.epochs is None and state["stale"] >= args.patience:
            break
        epoch = state["epoch"] + 1
        epoch_start = time.monotonic()
        skip = state["buffers"]
        seen = filled = blocks = train_targets = fed = 0
        val_rng = np.random.default_rng(148)
        print(f"epoch {epoch}, replaying {args.dataset}, {skip} completed buffers to skip", flush=True)

        def update(block, rows, targets):
            rng = np.random.default_rng([args.seed, epoch, block])
            train_buffer(model, optimizer, rows, rng, args)
            state["buffers"] = block
            if block % args.save_every == 0:
                save()
                print(f"epoch {epoch}, buffer {block}, {targets:,} scored training targets", flush=True)

        # One optimizer worker at most. While it owns one buffer the reader fills
        # the other. Join before reusing a buffer, validating, or releasing the run lock.
        pending = None

        def consume(n):
            nonlocal blocks, buffer, pending
            blocks += 1
            if blocks > skip:
                if pending is not None:
                    pending.result()
                if args.prefetch:
                    pending = worker.submit(update, blocks, buffer[:n], train_targets)
                else:
                    update(blocks, buffer[:n], train_targets)
            buffer = buffers[blocks % len(buffers)]

        buffer = buffers[0]
        with ThreadPoolExecutor(max_workers=1) as worker, source(args) as stream:
            for doc_id, doc in enumerate(documents(stream)):
                fed += len(doc)
                valid = held_out(doc_id)
                for crop in windows(doc, args.context, args.window):
                    if valid:
                        # Reservoir sampling fixes the validation set across all epochs and resumes.
                        seen += 1
                        slot = seen-1 if seen <= len(panel) else int(val_rng.integers(seen))
                        if slot < len(panel):
                            panel[slot] = crop
                    else:
                        train_targets += int(np.count_nonzero(crop["target"] != MASK))
                        buffer[filled] = crop
                        filled += 1
                        if filled == len(buffer):
                            consume(filled)
                            filled = 0
            if filled:
                consume(filled)
            if pending is not None:
                pending.result()
        if not train_targets or skip > blocks:
            raise ValueError("empty or shortened training stream")
        if not args.rows_file and args.dataset == "enwik8" and fed != 100_000_000:
            raise ValueError(f"enwik8 produced {fed:,} rows, expected 100,000,000")
        if state["history"] and (fed, train_targets) != (
                state["history"][0]["fed_rows"], state["history"][0]["train_targets"]):
            raise ValueError("replaying the same corpus produced different rows or targets")
        selected = panel[:min(seen, len(panel))]
        panel_sha = hashlib.sha256(selected).hexdigest()
        if state.get("panel_sha", panel_sha) != panel_sha:
            raise ValueError("validation inputs differ from the previous pass")
        state["panel_sha"] = panel_sha
        loss, n = validation(model, selected, args)
        if loss < state["best"]:
            state["best"] = loss
            state["best_epoch"] = epoch
            state["best_weights"] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if loss < state["anchor"] - args.min_delta:
            state.update(anchor=loss, stale=0)
        else:
            state["stale"] += 1
        state.update(epoch=epoch, buffers=0)
        state["history"].append(dict(epoch=epoch, fed_rows=fed, train_targets=train_targets,
                                     validation_targets=n, validation_bpb=loss))
        save()
        if state["best_epoch"] == epoch:
            export(model, dims, args.window, args.run_dir/"best.rrb")
        progress = (f"target epoch {args.epochs}" if args.epochs is not None else
                    f"patience {state['stale']}/{args.patience}")
        print(f"epoch {epoch}: {loss:.6f} validation bits/byte on {n:,} targets, "
              f"best epoch {state['best_epoch']}, {progress}, "
              f"{(time.monotonic()-epoch_start)/60:.2f} minutes elapsed", flush=True)
    if state.get("best_weights"):
        model.load_state_dict(state["best_weights"])
        export(model, dims, args.window, args.run_dir/"best.rrb")
    reason = "validation plateau" if args.epochs is None and state["stale"] >= args.patience else "epoch limit"
    print(json.dumps(dict(stopped=reason, best_epoch=state.get("best_epoch"), history=state["history"])), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    data = p.add_mutually_exclusive_group(required=True)
    data.add_argument("--work", help="Linux path prepared by live_data.py")
    data.add_argument("--rows-file", type=Path, help="finite RRL1 test fixture")
    p.add_argument("--dataset", choices=("enwik8", "enwik9"), default="enwik9")
    p.add_argument("--wsl", help="WSL distribution when training with Windows PyTorch")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--preset", choices=("compact", "compact-2x"), default="compact")
    p.add_argument("--score-feedback", action="store_true", help="revise the PPMd byte scores at every recursive update")
    p.add_argument("--prefetch", action=argparse.BooleanOptionalAction, default=True,
                   help="prepare the next CPU buffer during the current optimizer updates")
    p.add_argument("--cache-gib", type=int, default=0, help="maximum Linux disk space for reusable PPMd rows")
    p.add_argument("--context", type=int, default=128)
    p.add_argument("--window", type=int, default=64)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--buffer-windows", type=int, default=512)
    p.add_argument("--eval-windows", type=int, default=1024)
    p.add_argument("--save-every", type=int, default=32, help="training buffers per mid-epoch checkpoint")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--patience", type=int, default=3)
    p.add_argument("--min-delta", type=float, default=0.005, help="minimum validation improvement in bits/byte")
    duration = p.add_mutually_exclusive_group()
    duration.add_argument("--max-epochs", type=int, help="total epoch limit with validation early stopping")
    duration.add_argument("--epochs", type=int, help="train to this total epoch count even if validation plateaus")
    args = p.parse_args()
    if args.cache_gib < 0 or (args.rows_file and args.cache_gib):
        p.error("--cache-gib requires a nonnegative budget and live --work input")
    if not (5 <= args.context <= 2048 and 1 <= args.window <= args.context):
        p.error("context must be 5..2048 and window must be 1..context")
    if min(args.batch, args.buffer_windows, args.eval_windows, args.save_every, args.patience, args.threads) < 1:
        p.error("batch sizes and checkpoint/validation intervals must be positive")
    if args.min_delta < 0 or not math.isfinite(args.min_delta) or not 0 < args.lr < math.inf or args.seed < 0:
        p.error("invalid minimum improvement, learning rate or seed")
    if args.max_epochs is not None and args.max_epochs < 1:
        p.error("--max-epochs must be positive")
    if args.epochs is not None and args.epochs < 1:
        p.error("--epochs must be positive")
    if os.name == "nt" and args.work and not args.wsl:
        p.error("pass --wsl with the Ubuntu distribution name")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(args.run_dir/"run.lock", timeout=0):
        fit(args)


if __name__ == "__main__":
    main()
