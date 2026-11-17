"""Train the causal nested byte model and export it for src/refiner/."""
import argparse
import math
import struct
import sys
from functools import lru_cache
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parametrize

from recursive_refiner_hf import RecursiveRefinerConfig, RecursiveRefinerModel

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training/tf/2_package"))
import dumpv2

VOCAB = 208
INPUT_ROWS = 216
PRESETS = {
    # width, heads, FFN, shared blocks, high cycles, low cycles
    "smoke": (192, 3, 64, 1, 2, 1),
    "matched": (512, 8, 2298, 2, 2, 3),
}


def q4(weight):
    peak = weight.detach().abs().amax(dim=1, keepdim=True)
    scale = torch.where(peak > 0, peak / 7, 1.0)
    codes = (weight.detach() / scale).round().clamp(-7, 7)
    return codes, scale


class FakeQ4(torch.nn.Module):
    def forward(self, weight):
        codes, scale = q4(weight)
        return codes * scale + (weight - weight.detach())


def enable_q4(model):
    for layer in list(model.modules()):
        if isinstance(layer, (torch.nn.Linear, torch.nn.Embedding)):
            parametrize.register_parametrization(layer, "weight", FakeQ4())


def make_model(dims, window, max_positions):
    d, heads, ff, blocks, h, l = dims
    config = RecursiveRefinerConfig(vocab_size=INPUT_ROWS, hidden_size=d, num_attention_heads=heads,
        num_hidden_layers=blocks, expansion=ff/d, hi_cycles=h, lo_cycles=l, embed_factor=1,
        is_causal=True, max_position_embeddings=max(max_positions, 2*window+5))
    model = RecursiveRefinerModel(config)
    for block in model.shared.blocks:
        block.attn.window_size = window
    model.prior_gate = torch.nn.Parameter(torch.zeros(d))
    return model


def tensors_for_export(model, window):
    yield model.embed.tok
    yield from (model.hi_init, model.lo_init, model.prior_gate)
    rope = model.shared.blocks[0].attn.rope
    yield from rope(2*window)
    for b in model.shared.blocks:
        yield from (b.norm1.weight, b.norm2.weight)
        for proj in (b.attn.q_proj, b.attn.k_proj, b.attn.v_proj, b.attn.out, b.ffn.fc1, b.ffn.fc2):
            yield from (proj, proj.bias)


def export(model, dims, window, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(struct.pack("<4s7I", b"RRB2", *dims, window))
        for item in tensors_for_export(model, window):
            if isinstance(item, torch.nn.Module):
                codes, scale = q4(item.parametrizations.weight.original)
                tmp.write(scale.cpu().numpy().astype("<f4").tobytes())
                nibbles = (codes+7).to(torch.uint8).cpu().numpy()
                if nibbles.shape[1] % 2:
                    nibbles = np.pad(nibbles, ((0,0),(0,1)), constant_values=7)
                tmp.write((nibbles[:, ::2] | (nibbles[:, 1::2] << 4)).tobytes())
            else:
                tmp.write(item.detach().float().cpu().numpy().astype("<f4", copy=False).tobytes())
    Path(tmp.name).replace(path)


def demo_docs(path):
    docs, priors = [], []
    for name in ("train", "valid"):
        ids = np.fromfile(path / f"{name}.ids", dtype="u1")
        rows = np.fromfile(path / f"{name}.f16", dtype="<f2").reshape(-1, VOCAB)
        if len(rows) != len(ids) or len(ids) < 2 or ids.max() >= VOCAB:
            raise ValueError("invalid profiling IDs or prior trace")
        docs.append((ids[:-1], ids[1:], np.arange(len(ids)-1) + sum(map(len, priors))))
        priors.append(rows)
    return (docs.__getitem__, np.array([len(d[0]) for d in docs]), 1,
            np.concatenate(priors), np.zeros(VOCAB, np.float16))


def package_docs(path):
    rows, header, _ = dumpv2.lade(str(path / "zeilen.dump"))
    starts = np.fromfile(path / "doc-rows.int64", "<i8")
    last = np.fromfile(path / "doc-last-ziel.int32", "<i4")
    stream = np.memmap(path / "ppmd-bytes.uint8", dtype="u1", mode="r")
    priors = np.memmap(path / "ppmd-probs.float16", dtype="<f2", mode="r").reshape(-1, VOCAB)
    prior0 = np.fromfile(path / "z3-prior0.float16", dtype="<f2")
    if len(priors) != len(stream) or len(prior0) != VOCAB:
        raise ValueError("PPMd prior dimensions disagree with the token stream")
    lengths = np.diff(starts)
    if (not header or len(last) < 2 or len(last) != len(lengths)
            or starts[0] != 0 or starts[-1] != len(rows) or min(lengths) < 1):
        raise ValueError("expected the completed upstream stage-C package with at least two documents")

    @lru_cache(maxsize=32)
    def document(i):
        fields = dumpv2.felder(rows[starts[i]:starts[i+1]])
        positions, target, scored, _ = dumpv2.zielspalten(
            fields, np.array([0, lengths[i]]), VOCAB, last[i:i+1], stream)
        tokens = fields["tok"]
        if tokens.max() >= INPUT_ROWS or target[scored].max(initial=0) >= VOCAB:
            raise ValueError(f"invalid token in document {i}")
        prior_rows = dumpv2.priorziel(fields, positions) - 1
        return tokens, np.where(scored, target, -100), prior_rows

    split = max(1, min(len(last)-1, int(len(last)*0.9)))
    return document, lengths, split, priors, prior0


class Blocks:
    """Sample target spans with a left prefix from the same document."""
    def __init__(self, document, lengths, first_doc, context, window, priors, prior0):
        self.document = document
        self.warmup = min(window, context//2)
        self.stride = context - self.warmup
        self.ends = np.cumsum((lengths + self.stride - 1) // self.stride)
        self.priors, self.prior0 = priors, prior0
        self.first = first_doc
        self.context = context

    def batch(self, rng, size, device):
        x = np.zeros((size, self.context), dtype=np.int64)
        y = np.full_like(x, -100)
        p = np.zeros((*x.shape, VOCAB), dtype=np.float16)
        for b, idx in enumerate(rng.integers(self.ends[-1], size=size)):
            doc = self.ends.searchsorted(idx, side="right")
            target_start = int(idx - (self.ends[doc-1] if doc else 0)) * self.stride
            start = max(0, target_start - self.warmup)
            tokens, targets, indices = self.document(self.first + doc)
            span = slice(start, target_start + self.stride)
            rows = indices[span]
            n = len(rows)
            x[b, :n], y[b, :n] = tokens[span], targets[span]
            y[b, :target_start-start] = -100
            p[b, :n] = self.priors[np.maximum(rows, 0)]
            p[b, :n][rows < 0] = self.prior0
        return tuple(torch.from_numpy(a).to(device) for a in (x, y, p))


def byte_head(model, hidden, prior):
    raw = F.linear(hidden, model.embed.tok.weight[:VOCAB])
    scaled = hidden * (hidden.square().mean(-1, keepdim=True) + 1e-5).rsqrt()
    gate = F.softplus((scaled @ model.prior_gate).unsqueeze(-1) + 0.5413248546129181)
    logp = prior.float().clamp_min(math.exp(-12)).log()
    return 15*(raw/15).tanh() + gate*(logp-logp.mean(-1, keepdim=True))


def logits(model, x, prior):
    with parametrize.cached():
        hidden = model(x, return_dict=True).last_hidden_state
        return byte_head(model, hidden, prior)


@torch.no_grad()
def evaluate(model, data, batch, device):
    model.eval()
    rng = np.random.default_rng(148)
    loss, count = 0., 0
    for _ in range(8):
        x, y, prior = data.batch(rng, batch, device)
        loss += F.cross_entropy(logits(model, x, prior).flatten(0, 1), y.flatten(), reduction="sum").item()
        count += (y != -100).sum().item()
    if not count:
        raise ValueError("no scored targets in validation sample")
    return loss / (count * math.log(2)), count


@torch.no_grad()
def parity_fixture(model, window, path):
    # Compare incremental C++ against this exact HF model run in parallel
    # over each segment. Include prompt-only tokens, skipped readouts, an
    # explicit document reset and multiple wraps of the attention window.
    rng = np.random.default_rng(301)
    sizes = [2*window+3, window+5]
    model = model.cpu().eval()
    with path.open("wb") as out:
        out.write(b"RRP2")
        for part, size in enumerate(sizes):
            ids = rng.integers(INPUT_ROWS, size=size, dtype=np.uint8)
            ids[0] = 215
            low = []
            def capture(_module, _inputs, result):
                low.append(result)
            handle = model.shared.register_forward_hook(capture)
            try:
                hidden = model(torch.tensor(ids).long()[None, :]).last_hidden_state[0]
            finally:
                handle.remove()
            prior = rng.random((size,VOCAB), dtype=np.float32)**4
            prior[:, 0] = 0
            prior = (prior/prior.sum(-1, keepdims=True)).astype("<f2").astype("<f4")
            probs = byte_head(model, hidden, torch.from_numpy(prior)).softmax(-1)
            for i, token in enumerate(ids):
                reset = part == 1 and i == 0
                read = i % 5 != 2
                out.write(bytes((int(token), int(reset), int(read))))
                out.write(prior[i].tobytes())
                for row in (probs[i], hidden[i, :192], low[-2][0, i, :192]):
                    out.write(row.numpy().astype("<f4", copy=False).tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, default="matched")
    p.add_argument("--blocks", type=int, help="number of distinct blocks in the shared stack")
    p.add_argument("--hi-cycles", type=int, help="high-state updates per token")
    p.add_argument("--lo-cycles", type=int, help="low-state updates per high-state update")
    p.add_argument("--ffn-width", type=int, help="SwiGLU intermediate width")
    p.add_argument("--package", type=Path, help="upstream stage-C directory (default: profiling demo)")
    p.add_argument("--demo-dir", type=Path, default=ROOT/"build-refiner/demo",
                   help="profiling IDs and PPMd rows written by prepare_demo.py")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--context", type=int, default=64, help="training crop length")
    p.add_argument("--window", type=int, help="causal attention window (default: half the training crop)")
    p.add_argument("--qat-steps", type=int, help="final steps with simulated Q4 weights (default: final quarter)")
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0002)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=Path("recursive-refiner.rrb"))
    p.add_argument("--parity", type=Path, help="write C++ parity fixture")
    p.add_argument("--sample", type=Path, help="save up to 4 KiB of held-out demo text for cmix")
    args = p.parse_args()
    if not 5 <= args.context <= 2048 or min(args.batch, args.threads, args.steps) < 1:
        p.error("context must be 5..2048 and batch, threads and steps must be positive")
    if args.package and args.sample:
        p.error("--sample is available for the profiling demo")
    window = args.context//2 if args.window is None else args.window
    qat_steps = max(1, args.steps//4) if args.qat_steps is None else args.qat_steps
    if not 1 <= window <= args.context or not 1 <= qat_steps <= args.steps:
        p.error("window must be 1..context and qat-steps must be 1..steps")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    d, heads, ff, blocks, h, l = PRESETS[args.preset]
    if args.ffn_width is not None:
        ff = args.ffn_width
    if args.blocks is not None:
        blocks = args.blocks
    if args.hi_cycles is not None:
        h = args.hi_cycles
    if args.lo_cycles is not None:
        l = args.lo_cycles
    if not (1 <= ff <= 8192 and 1 <= blocks <= 32 and 1 <= h <= 8 and 1 <= l <= 8):
        p.error("native model supports ffn-width 1..8192, blocks 1..32 and hi/lo cycles 1..8")
    if 2 * blocks * h * (l + 1) * d * window > 2**27:
        p.error("this configuration exceeds the native 512 MiB KV cache limit")
    dims = (d, heads, ff, blocks, h, l)
    model = make_model(dims, window, args.context).to(args.device)
    count = sum(t.numel() for t in model.parameters())
    print(f"{args.preset}: {count:,} trainable parameters; baseline 9,285,596", flush=True)
    if not args.package and not (args.demo_dir/"train.f16").exists():
        p.error("prepare profiling priors with training/refiner/prepare_demo.py, or specify --package")
    document, lengths, split, priors, prior0 = package_docs(args.package) if args.package else demo_docs(args.demo_dir)
    train = Blocks(document, lengths[:split], 0, args.context, window, priors, prior0)
    valid = Blocks(document, lengths[split:], split, args.context, window, priors, prior0)
    before, scored = evaluate(model, valid, args.batch, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    trained = 0
    for step in range(args.steps):
        if step == args.steps - qat_steps:
            enable_q4(model)
        for attempt in range(100):
            x, y, prior = train.batch(rng, args.batch, args.device)
            if (y != -100).any(): break
        else:
            raise ValueError("no scored targets in 100 sampled training batches")
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits(model, x, prior).flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        trained += (y != -100).sum().item()
    after, _ = evaluate(model, valid, args.batch, args.device)
    print(f"validation bits/scored byte: {before:.4f} -> {after:.4f} ({scored} sampled targets); "
          f"training targets: {trained}", flush=True)
    export(model, dims, window, args.out)
    if args.parity: parity_fixture(model, window, args.parity)
    if args.sample: args.sample.write_bytes((args.demo_dir/"heldout.bin").read_bytes())


if __name__ == "__main__":
    main()
