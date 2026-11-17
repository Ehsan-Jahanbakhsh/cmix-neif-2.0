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


def tensors_for_export(model, context):
    yield from (model.embed.tok.weight, model.hi_init, model.lo_init)
    rope = model.shared.blocks[0].attn.rope
    yield from rope(context)
    for b in model.shared.blocks:
        yield from (b.norm1.weight, b.norm2.weight)
        for proj in (b.attn.q_proj, b.attn.k_proj, b.attn.v_proj, b.attn.out, b.ffn.fc1, b.ffn.fc2):
            yield from (proj.weight, proj.bias)


def export(model, dims, context, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(struct.pack("<4s7I", b"RRB1", *dims, context))
        for tensor in tensors_for_export(model, context):
            tmp.write(tensor.detach().float().cpu().numpy().astype("<f4", copy=False).tobytes())
    Path(tmp.name).replace(path)


def demo_docs():
    # Profiling text is distributed with upstream. Map only bytes that cmix can
    # code. The split is made before any context windows are constructed.
    bitmap = (ROOT / "training/tf/2_package/vocab-bitmap.u8").read_bytes()
    alphabet = bytes(i for i in range(256) if bitmap[i // 8] & (1 << (i % 8)))
    assert len(alphabet) == VOCAB
    lut = {b: i for i, b in enumerate(alphabet)}
    raw = (ROOT / "prof_input/input").read_bytes().lower()
    raw = bytes(b for b in raw if b in lut)
    split = len(raw) * 9 // 10
    docs = [np.fromiter((lut[b] for b in part), np.uint8) for part in (raw[:split], raw[split:])]

    def document(i):
        return docs[i][:-1], docs[i][1:]

    return document, np.array([len(d)-1 for d in docs]), 1, raw[split:]


def package_docs(path):
    rows, header, _ = dumpv2.lade(str(path / "zeilen.dump"))
    starts = np.fromfile(path / "doc-rows.int64", "<i8")
    last = np.fromfile(path / "doc-last-ziel.int32", "<i4")
    stream = np.memmap(path / "ppmd-bytes.uint8", dtype="u1", mode="r")
    lengths = np.diff(starts)
    if (not header or len(last) < 2 or len(last) != len(lengths)
            or starts[0] != 0 or starts[-1] != len(rows) or min(lengths) < 1):
        raise ValueError("expected the completed upstream stage-C package with at least two documents")

    @lru_cache(maxsize=32)
    def document(i):
        fields = dumpv2.felder(rows[starts[i]:starts[i+1]])
        _, target, scored, _ = dumpv2.zielspalten(
            fields, np.array([0, lengths[i]]), VOCAB, last[i:i+1], stream)
        tokens = fields["tok"]
        if tokens.max() >= INPUT_ROWS or target[scored].max(initial=0) >= VOCAB:
            raise ValueError(f"invalid token in document {i}")
        return tokens, np.where(scored, target, -100)

    split = max(1, min(len(last)-1, int(len(last)*0.9)))
    return document, lengths, split, None


class Blocks:
    """Uniform sampling over non-overlapping runtime context blocks."""
    def __init__(self, document, lengths, first_doc, context):
        self.document = document
        self.ends = np.cumsum((lengths + context - 1) // context)
        self.first = first_doc
        self.context = context

    def batch(self, rng, size, device):
        x = np.zeros((size, self.context), dtype=np.int64)
        y = np.full_like(x, -100)
        for b, idx in enumerate(rng.integers(self.ends[-1], size=size)):
            doc = self.ends.searchsorted(idx, side="right")
            start = int(idx - (self.ends[doc-1] if doc else 0)) * self.context
            tokens, targets = self.document(self.first + doc)
            window = slice(start, start + self.context)
            n = len(tokens[window])
            x[b, :n], y[b, :n] = tokens[window], targets[window]
        return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def logits(model, x):
    hidden = model(x, return_dict=True).last_hidden_state
    return F.linear(hidden, model.embed.tok.weight[:VOCAB])


@torch.no_grad()
def evaluate(model, data, batch, device):
    model.eval()
    rng = np.random.default_rng(148)
    loss, count = 0., 0
    for _ in range(8):
        x, y = data.batch(rng, batch, device)
        loss += F.cross_entropy(logits(model, x).flatten(0, 1), y.flatten(), reduction="sum").item()
        count += (y != -100).sum().item()
    if not count:
        raise ValueError("no scored targets in validation sample")
    return loss / (count * math.log(2)), count


@torch.no_grad()
def parity_fixture(model, context, path):
    # Compare incremental C++ against this exact HF model run in parallel
    # over each segment. Include prompt-only tokens, skipped readouts, an
    # explicit document reset and an automatic context reset.
    rng = np.random.default_rng(301)
    sizes = [5, context, 3]
    model = model.cpu().eval()
    with path.open("wb") as out:
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
            probs = F.linear(hidden, model.embed.tok.weight[:VOCAB]).softmax(-1)
            for i, token in enumerate(ids):
                reset = part == 1 and i == 0
                read = i % 5 != 2
                out.write(bytes((int(token), int(reset), int(read))))
                for row in (probs[i], hidden[i, :192], low[-2][0, i, :192]):
                    out.write(row.numpy().astype("<f4", copy=False).tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preset", choices=PRESETS, default="matched")
    p.add_argument("--package", type=Path, help="upstream stage-C directory (default: profiling demo)")
    p.add_argument("--steps", type=int, default=64)
    p.add_argument("--context", type=int, default=32)
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
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dims = PRESETS[args.preset]
    d, heads, ff, blocks, h, l = dims
    config = RecursiveRefinerConfig(vocab_size=INPUT_ROWS, hidden_size=d, num_attention_heads=heads,
        num_hidden_layers=blocks, expansion=ff/d, hi_cycles=h, lo_cycles=l, embed_factor=1,
        is_causal=True, max_position_embeddings=args.context)
    model = RecursiveRefinerModel(config).to(args.device)
    count = sum(t.numel() for t in model.parameters())
    print(f"{args.preset}: {count:,} trainable parameters; baseline 9,285,596", flush=True)
    document, lengths, split, sample = package_docs(args.package) if args.package else demo_docs()
    train = Blocks(document, lengths[:split], 0, args.context)
    valid = Blocks(document, lengths[split:], split, args.context)
    before, scored = evaluate(model, valid, args.batch, args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    trained = 0
    for _ in range(args.steps):
        for attempt in range(100):
            x, y = train.batch(rng, args.batch, args.device)
            if (y != -100).any(): break
        else:
            raise ValueError("no scored targets in 100 sampled training batches")
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(logits(model, x).flatten(0, 1), y.flatten())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
        trained += (y != -100).sum().item()
    after, _ = evaluate(model, valid, args.batch, args.device)
    print(f"validation bits/scored byte: {before:.4f} -> {after:.4f} ({scored} sampled targets); "
          f"training targets: {trained}", flush=True)
    export(model, dims, args.context, args.out)
    if args.parity: parity_fixture(model, args.context, args.parity)
    if args.sample: args.sample.write_bytes(sample[:4096])


if __name__ == "__main__":
    main()
