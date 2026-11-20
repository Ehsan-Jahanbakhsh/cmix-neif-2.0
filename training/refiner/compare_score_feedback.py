"""Time paired Q4 refiner training runs on complete documents from the enwik9 row cache."""
import argparse
import hashlib
import json
import math
import time
import zlib
from pathlib import Path

import numpy as np
import torch

from live_data import CHUNK_ROWS
from train import PRESETS, enable_q4, export, make_model
from train_live import MASK, ROW, atomic_save, batch_loss, validation, windows


def chunk_windows(path, args):
    rows = np.frombuffer(zlib.decompress(path.read_bytes()), ROW)
    if len(rows) != CHUNK_ROWS:
        raise ValueError(f"incomplete row-cache chunk {path}")
    starts = np.flatnonzero(rows["start"])
    for left, right in zip(starts, starts[1:]):
        yield from windows(rows[left:right], args.context, args.window)


def training_batches(paths, args, rng, manifest):
    while True:
        buffer = []
        used = 0
        emitted = False
        for idx in rng.permutation(len(paths)):
            path = paths[idx]
            try:
                crops = list(chunk_windows(path, args))
            except FileNotFoundError:
                continue  # The live cache can evict old chunks.
            if crops:
                manifest["training_chunk_visits"].append(int(path.stem))
                used += 1
                buffer.extend(crops)
            if len(buffer) >= 4 * args.batch:
                rng.shuffle(buffer)
                cutoff = len(buffer) // args.batch * args.batch
                for start in range(0, cutoff, args.batch):
                    emitted = True
                    yield np.stack(buffer[start:start + args.batch])
                buffer = buffer[cutoff:]
        if not used:
            raise RuntimeError("No complete training documents remain in the selected cache chunks")
        if not emitted:
            if len(buffer) < args.batch:
                raise ValueError("batch exceeds the available training windows")
            rng.shuffle(buffer)
            for start in range(0, len(buffer)-args.batch+1, args.batch):
                yield np.stack(buffer[start:start + args.batch])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds-per-model", type=float, default=900)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    if (not 0 < args.seconds_per_model < math.inf or args.batch < 1
            or not 1 <= args.window <= args.context <= 2048):
        parser.error("invalid budget, batch size or window")
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    rng = np.random.default_rng(148)
    # Skip the two newest files, which may still be open for writing.
    paths = sorted(args.cache.glob("*.zlib"), key=lambda p: int(p.stem))[-1026:-2]
    if len(paths) < 32:
        raise ValueError("need at least 32 complete cache chunks")
    eval_ids = set(rng.choice(len(paths), size=min(32, len(paths)//4), replace=False).tolist())
    panel, groups = [], []
    for i in sorted(eval_ids):
        crops = list(chunk_windows(paths[i], args))
        if crops:
            selected = rng.choice(len(crops), size=min(32, len(crops)), replace=False)
            panel.extend(crops[j] for j in selected)
            groups.extend([int(paths[i].stem)] * len(selected))
    if not panel:
        raise ValueError("no complete held-out documents")
    panel = np.stack(panel)
    np.savez_compressed(args.out/"heldout.npz", panel=panel, groups=np.array(groups))
    manifest = dict(candidate_offsets=[int(p.stem) for p in paths],
                    validation_offsets=[int(paths[i].stem) for i in sorted(eval_ids)],
                    panel_sha256=hashlib.sha256(panel).hexdigest(), training_chunk_visits=[])
    batches = training_batches([p for i, p in enumerate(paths) if i not in eval_ids], args,
                               np.random.default_rng(1), manifest)
    names = ["current", "score-feedback"]
    dims = PRESETS["compact-2x"]
    models, optimizers, stats = {}, {}, {}
    for name in names:
        torch.manual_seed(1)
        model = make_model(dims, args.window, args.context, name == "score-feedback").to(args.device)
        enable_q4(model)
        models[name] = model
        optimizers[name] = torch.optim.AdamW(model.parameters(), lr=2e-4)
        stats[name] = dict(seconds=0., steps=0, targets=0, train_nats=0.)
    report = dict(seconds_per_model=args.seconds_per_model, batch=args.batch, context=args.context,
                  window=args.window, device=args.device, panel_windows=len(panel),
                  seed=1, optimizer="AdamW", lr=2e-4, preset="compact-2x", snapshots={})

    def snapshot(label):
        rows = {}
        for name, model in models.items():
            bpb, count = validation(model, panel, args)
            item = dict(stats[name], heldout_bpb=bpb, heldout_targets=count,
                        parameters=sum(p.numel() for p in model.parameters()))
            if item["targets"]:
                item["online_train_bpb"] = item["train_nats"] / (math.log(2)*item["targets"])
            rows[name] = item
        report["snapshots"][label] = rows
        (args.out/"results.json").write_text(json.dumps(report, indent=2))
        (args.out/"samples.json").write_text(json.dumps(manifest))
        print(label, json.dumps(rows), flush=True)

    snapshot("initial")
    # Alternating which model goes first limits GPU clock and background-load bias.
    # Both train on identical batches until the slower model hits its time budget.
    for turn, batch in enumerate(batches):
        count = int(np.count_nonzero(batch["target"] != MASK))
        for name in (names if turn % 2 == 0 else names[::-1]):
            st = stats[name]
            if st["seconds"] >= args.seconds_per_model:
                continue
            model, opt = models[name], optimizers[name]
            if args.device == "cuda":
                torch.cuda.synchronize()
            start = time.monotonic()
            opt.zero_grad(set_to_none=True)
            loss, _ = batch_loss(model, batch, args.device)
            (loss/count).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step()
            st["train_nats"] += loss.item()
            if args.device == "cuda":
                torch.cuda.synchronize()
            st["seconds"] += time.monotonic() - start
            st["steps"] += 1
            st["targets"] += count
        done = [stats[n]["seconds"] >= args.seconds_per_model for n in names]
        if any(done) and "equal_updates" not in report["snapshots"]:
            snapshot("equal_updates")
            matched = args.out/"equal-updates"
            matched.mkdir()
            for name, model in models.items():
                export(model, dims, args.window, matched/f"{name}.rrb")
        if all(done):
            break
        if (turn+1) % 250 == 0:
            print("progress", {n: (s["steps"], round(s["seconds"], 1)) for n, s in stats.items()}, flush=True)
    snapshot("equal_time")
    for name, model in models.items():
        atomic_save(dict(model=model.state_dict(), dims=dims, window=args.window,
                         score_feedback=model.score_feedback), args.out/f"{name}.pt")
        export(model, dims, args.window, args.out/f"{name}.rrb")


if __name__ == "__main__":
    main()
