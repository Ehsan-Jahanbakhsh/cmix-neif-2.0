"""Train the Recursive Refiner on the original cmix enwik9 package and epoch schedule."""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from filelock import FileLock

from train import PRESETS, ROOT, byte_head, enable_q4, export, logits, make_model, package_docs

TF = ROOT / "training/tf"
sys.path[:0] = [str(TF / "tree_overlay"), str(TF / "tree_upstream")]
from zctx.zpdump import PaketDumpDataLoader


STAGES = ("b1", "b2", "b3", "qat", "fp32")
FILES = ("01_b1_bf16.json", "02_b2_bf16.json", "03_b3_bf16_epochs_1_to_29.json",
         "06_qat.json", "07_fp32.json")
CHAIN = TF / "3_train/chain"
ENWIK9_SHA = "159b85351e5f76e60cbe32e04c677847a9ecba3adc79addab6f4c6c7aa3744bc"


class BatchPlan:
    # Use the upstream loader's two grouping methods directly. A rank reads
    # only its assigned documents. Our byte head uses their tokens and priors.
    _iter_micro_batch_articles = PaketDumpDataLoader._iter_micro_batch_articles
    groups = PaketDumpDataLoader._iter_batch_article_groups

    def __init__(self, lengths, micro_rows, update_rows, world):
        self._doc_rows = np.r_[0, lengths.cumsum()]
        self.micro_batch_tokens = micro_rows
        self.world_size = world
        self.gradient_accumulation_steps = update_rows // (micro_rows * world)
        self.n = len(lengths)

    def orders(self, epochs):
        rng = np.random.default_rng(0)
        for _ in range(epochs):
            yield rng.permutation(self.n)


def padded_batches(articles, lengths, limit):
    # Group similar lengths to bound padding. Every document keeps all of its
    # causal history. Right padding never affects a scored token.
    group = []
    for i in sorted(articles, key=lengths.__getitem__):
        if group and (len(group) + 1) * lengths[i] > limit:
            yield group
            group = []
        group.append(i)
    if group:
        yield group


def examples(articles, document, lengths, priors, prior0, device):
    width = max(lengths[i] for i in articles)
    x = np.zeros((len(articles), width), dtype=np.int64)
    y = np.full_like(x, -100)
    p = np.zeros((*x.shape, 208), dtype=np.float16)
    for b, i in enumerate(articles):
        tok, targets, rows = document(i)
        n = len(tok)
        x[b, :n], y[b, :n] = tok, targets
        p[b, :n] = priors[np.maximum(rows, 0)]
        p[b, :n][rows < 0] = prior0
    return (torch.from_numpy(a).to(device) for a in (x, y, p))


def document_loss(model, batch, use_bf16, checkpoint_from_rows):
    x, y, prior = batch
    model.shared.checkpoint_blocks = x.numel() >= checkpoint_from_rows
    with torch.autocast(x.device.type, dtype=torch.bfloat16, enabled=use_bf16):
        if model.score_feedback:
            scores = logits(model, x, prior)
        else:
            high = model(x).last_hidden_state
            scores = byte_head(model, high, prior)
    return F.cross_entropy(scores.float().flatten(0, 1), y.flatten(), reduction="sum"), (y != -100).sum()


def optimizers(model):
    matrices = [p for p in model.shared.parameters() if p.ndim == 2]
    matrix_ids = {id(p) for p in matrices}
    rest = [p for p in model.parameters() if id(p) not in matrix_ids]
    return (torch.optim.Muon(matrices, lr=5e-4, weight_decay=0.1, momentum=0.95,
                             adjust_lr_fn="match_rms_adamw"),
            torch.optim.AdamW(rest, lr=5e-4, weight_decay=0.01, betas=(0.95, 0.99)))


def cosine(start, end, fraction):
    return end + (start - end) * (1 + math.cos(math.pi * fraction)) / 2


def learning_rate(cfg, step, total):
    f = (step + 1) / total
    if "rewarm" in cfg:
        r = cfg["rewarm"]
        if f < r["frac"]:
            return cfg["lr"] * cosine(r["start"], r["peak"], f/r["frac"])
        return cfg["lr"] * cosine(r["peak"], r["final"], (f-r["frac"])/(1-r["frac"]))
    if "cosine_exact" in cfg:
        f = step / max(1, total-1)
    a = cfg.get("anneal", cfg.get("cosine_exact", {}))
    return cfg["lr"] * cosine(a.get("from_mult", a.get("from", 1)),
                              a.get("to_mult", a.get("to", cfg.get("final_mult", 1))), f)


def save(path, state):
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        name = Path(tmp.name)
    try:
        torch.save(state, name)
        name.replace(path)
    finally:
        name.unlink(missing_ok=True)


def reduce_gradients(model, world):
    if world == 1:
        return
    # One collective per update even when ranks use different numbers of
    # padded forwards. All ranks use the same active parameters.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat)
    flat.div_(world)
    for grad, values in zip(grads, flat.split([g.numel() for g in grads])):
        grad.copy_(values.view_as(grad))


def run(args, device, rank, world):
    configs = [json.loads((CHAIN / f).read_text()) for f in FILES]
    b3_cont = json.loads((CHAIN / "03_b3_bf16_epochs_30_to_400.json").read_text())
    epochs = [1 if args.smoke else c["epochs"] for c in configs]
    window = configs[0]["overrides"]["window_size"]
    dims = PRESETS[args.preset or ("smoke" if args.smoke else "matched")]
    if not args.smoke:
        if (args.package/"ppmd-probs.float16").stat().st_size != 203023960672:
            raise ValueError("full run needs the original stage-C enwik9 package (203023960672 prior bytes)")
        manifest = args.package / "MANIFEST.json"
        if manifest.exists():
            source = json.loads(manifest.read_text(encoding="utf-8"))["eingaenge"]["enwik9"]
            if source["sha256"] != ENWIK9_SHA:
                raise ValueError("stage-C manifest identifies a different input from the original enwik9")
    document, lengths, _, priors, prior0 = package_docs(args.package)
    if lengths.max() > args.micro_rows:
        raise ValueError("the longest document exceeds --micro-rows")
    plan = BatchPlan(lengths, args.micro_rows, args.update_rows, world)
    signature = (dims, window, tuple(epochs), args.micro_rows, args.update_rows, args.forward_rows,
                 hashlib.sha256(lengths.tobytes()).hexdigest(), len(priors), configs, b3_cont)
    if args.score_feedback:
        signature += ("score-feedback-v1",)
    state_path = args.run_dir / "state.pt"
    previous = torch.load(state_path, map_location="cpu", weights_only=True) if state_path.exists() else None
    if previous and previous["plan"] != signature:
        raise ValueError("run settings or package differ from the saved run")
    torch.manual_seed(configs[0]["seed"])
    model = make_model(dims, window, int(lengths.max()), args.score_feedback).to(device)
    checkpoint_from_rows = args.checkpoint_from_rows
    if checkpoint_from_rows is None:
        checkpoint_from_rows = 8193 if args.preset == "compact" else 1
    for block in model.shared.blocks:
        block.attn.whole_documents = True
    if rank == 0:
        print(f"{sum(p.numel() for p in model.parameters()):,} parameters, window {window}, "
              f"{len(lengths):,} documents, {world} rank(s), {plan.gradient_accumulation_steps} "
              f"microbatches per rank per update", flush=True)
    opts = optimizers(model)
    first, done = (previous["stage"], previous["epoch"]) if previous else (0, 0)
    if first > STAGES.index(args.until_stage):
        if rank == 0:
            print(f"The saved run has already passed {args.until_stage}.")
        return
    quantized = first >= STAGES.index("qat")
    if quantized:
        enable_q4(model)
    if previous:
        model.load_state_dict(previous["model"])
        for opt, data in zip(opts, previous["optimizers"]):
            opt.load_state_dict(data)
        del previous
    counts = []
    for stage in range(first, len(STAGES)):
        name, cfg, horizon = STAGES[stage], configs[stage], epochs[stage]
        if done < horizon:
            if stage >= STAGES.index("qat") and not quantized:
                enable_q4(model)
                quantized = True
            if name == "fp32" and done == 0:
                # Upstream starts a fresh optimizer for this cooldown. Preserve
                # group order from before Q4 parametrizations were installed.
                for opt in opts:
                    opt.state.clear()
            if len(counts) < horizon:
                counts = [sum(1 for _ in plan.groups(order)) for order in plan.orders(horizon)]
            if min(counts[:horizon]) == 0:
                raise ValueError("too few documents for a complete optimizer update")
            total = sum(counts[:horizon])
            step = sum(counts[:done])
            if rank == 0:
                print(f"{name}: {horizon} epochs, {total} updates, resume epoch {done+1}", flush=True)
            for epoch, order in enumerate(plan.orders(horizon)):
                if epoch < done:
                    continue
                stats = torch.zeros(2, device=device, dtype=torch.float64)
                # reproduce.sh switches the b3 configuration after epoch 29.
                schedule = b3_cont if name == "b3" and epoch >= 29 else cfg
                for groups in plan.groups(order):
                    lr = learning_rate(schedule, step, total)
                    for opt in opts:
                        opt.zero_grad(set_to_none=True)
                        for group in opt.param_groups:
                            group["lr"] = lr
                    for ids in groups[rank::world]:
                        for chunk in padded_batches(ids, lengths, args.forward_rows):
                            batch = examples(chunk, document, lengths, priors, prior0, device)
                            loss, n = document_loss(model, batch, name != "fp32", checkpoint_from_rows)
                            (loss / plan.gradient_accumulation_steps).backward()
                            stats += torch.stack((loss.detach(), n)).double()
                    reduce_gradients(model, world)
                    for opt in opts:
                        opt.step()
                    step += 1
                if world > 1:
                    dist.all_reduce(stats)
                nats, n = stats.tolist()
                if not (math.isfinite(nats) and n > 0):
                    raise ValueError("non-finite training loss or no scored targets")
                if rank == 0:
                    save(state_path, dict(plan=signature, stage=stage, epoch=epoch+1, model=model.state_dict(),
                                          optimizers=[opt.state_dict() for opt in opts]))
                    print(f"{name} epoch {epoch+1}: {int(n)} scored rows, {nats/n/math.log(2):.6f} training bits/byte",
                          flush=True)
                if world > 1:
                    dist.barrier()
        if name == args.until_stage and name != "fp32":
            return
        done = 0
    model.eval()
    stats = torch.zeros(2, device=device, dtype=torch.float64)
    with torch.no_grad():
        for i, ids in enumerate(plan._iter_micro_batch_articles(range(len(lengths)), drop_last=False)):
            if i % world == rank:
                for chunk in padded_batches(ids, lengths, args.forward_rows):
                    batch = examples(chunk, document, lengths, priors, prior0, device)
                    loss, n = document_loss(model, batch, False, checkpoint_from_rows)
                    stats += torch.stack((loss, n)).double()
    if world > 1:
        dist.all_reduce(stats)
    nats, n = stats.tolist()
    if not math.isfinite(nats) or not n or (not args.smoke and n != 487298491):
        raise ValueError(f"invalid full-package evaluation: {n} scored rows, {nats} nats")
    if rank == 0:
        path = args.run_dir / "recursive-refiner.rrb"
        export(model, dims, window, path)
        print(f"{int(n)} scored rows, {nats/n/math.log(2):.6f} in-sample bits/byte, {path}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--package", type=Path, required=True, help="completed upstream pkg_check directory")
    p.add_argument("--run-dir", type=Path, required=True, help="checkpoint, resume state and export directory")
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--preset", choices=("compact", "compact-2x", "matched"), help="model size (default: matched for a full run)")
    p.add_argument("--score-feedback", action="store_true", help="revise the PPMd byte scores at every recursive update")
    p.add_argument("--until-stage", choices=STAGES, default="fp32")
    p.add_argument("--micro-rows", type=int, default=131072)
    p.add_argument("--update-rows", type=int, default=1048576)
    p.add_argument("--forward-rows", type=int, default=8192, help="padding budget per forward, whole documents retained")
    p.add_argument("--checkpoint-from-rows", type=int, help="recompute blocks during backward at or above this many "
                   "padded rows per forward (default: 8193 for compact, 1 otherwise)")
    p.add_argument("--smoke", action="store_true", help="small model and one epoch per stage on a test package")
    args = p.parse_args()
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    if min(args.forward_rows, args.micro_rows, args.update_rows) < 1 or args.update_rows % (args.micro_rows*world):
        p.error("row budgets must be positive and update-rows divisible by micro-rows times ranks")
    if args.checkpoint_from_rows is not None and args.checkpoint_from_rows < 1:
        p.error("checkpoint-from-rows must be positive")
    device = torch.device(args.device)
    if args.device == "cuda":
        device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", 0)))
        torch.cuda.set_device(device)
        if not torch.cuda.is_bf16_supported():
            p.error("this recipe requires BF16 support")
        from flash_attn import flash_attn_func  # check the whole-document CUDA kernel before training
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    try:
        with FileLock(args.run_dir/"run.lock", timeout=0) if rank == 0 else nullcontext():
            run(args, device, rank, world)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
