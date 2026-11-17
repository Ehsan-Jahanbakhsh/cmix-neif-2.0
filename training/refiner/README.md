# Recursive Refiner in cmix

You can train, export and score the byte predictor before running any compression. The refiner learns to predict the next byte from preceding tokens and cmix's PPMd distribution. The C++ implementation uses the exported weights for both encoding and decoding. [Python blocks](recursive_refiner_hf.py) · [Trainer](train.py) · [C++ inference](../../src/refiner/refiner.cpp)

## Build the CLI tools

Run these commands from the fork root in Ubuntu 24.04 on an x86-64 CPU with AVX2 and F16C. Ubuntu under WSL2 works. Use a Linux filesystem for cmix's scratch files.

```bash
sudo apt-get update
sudo apt-get install -y clang-18 lld-18 make g++ python3-venv
mkdir -p build-refiner
make -j4 CC=clang++-18 BYTE_MODEL=refiner SAMPLE=1 cmix
cp cmix build-refiner/cmix-refiner
make -j4 CC=clang++-18 BYTE_MODEL=transformer SAMPLE=1 cmix
cp cmix build-refiner/cmix-transformer
cp weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5 build-refiner/
clang++-18 -std=c++17 -O3 -march=x86-64-v3 -ffp-model=precise -ffp-contract=off \
  -ffunction-sections -fdata-sections -Wl,--gc-sections training/refiner/check.cpp \
  src/refiner/refiner.cpp src/tf1/opt/glue.cpp src/detmath.cpp -o build-refiner/refiner-check
```

## Train the refiner separately

The first command below the install steps prepares the bundled profiling text. It lowercases the bytes, filters to cmix's frozen alphabet, reserves the final 10% for validation and runs PPMd independently on each split. Each retained byte takes 416 bytes of prior data on disk. For a small text corpus of your own, append `--input my-text.txt` to the preparation command. Once the priors exist, the trainer runs independently of cmix.

```bash
python3 -m venv build-refiner/venv
source build-refiner/venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'transformers>=4.45,<5' numpy

python training/refiner/prepare_demo.py build-refiner/cmix-refiner build-refiner/demo
python training/refiner/train.py --preset matched --steps 128 --qat-steps 32 \
  --context 128 --window 64 --threads 6 --demo-dir build-refiner/demo \
  --out build-refiner/recursive-refiner.rrb --parity build-refiner/parity.bin \
  --sample build-refiner/heldout.bin
build-refiner/refiner-check build-refiner/recursive-refiner.rrb build-refiner/parity.bin
python training/refiner/benchmark.py build-refiner --score-only
```

The fork includes the trained demo checkpoint as `recursive-refiner.rrb` at the repository root. To score those weights directly after compiling, copy this file and `build-refiner/demo/heldout.bin` into `build-refiner/` after the preparation command.

`--blocks` controls the number of distinct blocks in the shared stack. `--hi-cycles` and `--lo-cycles` control how many times the same weights are used. Changing just the cycles preserves the parameter budget and changes the compute per token. `--ffn-width` changes the SwiGLU width and the parameter count. `--context` sets the training crop length and `--window` sets the attention window for training and inference. Each crop uses up to half its length for left context, capped at the window length. The trainer excludes those warmup tokens from the loss.

The matched preset has width 512, eight heads, two shared blocks, FFN width 2,298, two high cycles and three low updates per high cycle. This gives 16 block evaluations per token and **9,285,096 trainable parameters**, 500 fewer than the upstream transformer. Each pass keeps a separate rolling KV history until a document reset. The first 192 high-state and low-state features feed cmix's existing BITLSTM and semantic cache. The byte head ties the embedding to the output projection and learns a gate on the PPMd log probabilities. The upstream transformer also has learned match features and an online second head. [Original Recursive Refiner](https://github.com/Ehsan-Jahanbakhsh/cramming-me/blob/15766a659deced4c18be5cc2625cbad3ebd56ae5/cramming/architectures/recursive_refiner_hf.py)

Training ends with simulated per-row 4-bit weights. The exporter packs two weights per byte for the embeddings and linear layers, with FP32 row scales. C++ expands the codes to signed bytes on load and accumulates in FP32. The matched 64-token-window checkpoint occupies **4,796,096 bytes**. Upstream's mixed-precision container occupies **3,744,629 bytes**.

## Results from the included training recipe

The CPU demo saw 16,384 scored training targets across 128 AdamW steps. Loss fell from 5.2502 to 2.0288 bits per scored byte over a fixed validation sample of 1,024 targets. On 200 streaming parity rows, including prompt tokens, skipped readouts, multiple window wraps and a document reset, the largest C++ versus PyTorch probability difference was 0.000000194.

On the same 4,096-byte held-out body excerpt, starting from a fresh predictor for each model, the C++ prediction-only CLI gave the following results over 4,095 next-byte predictions.

| Model | Bits per scored byte |
|---|---:|
| Shipped transformer | 2.079623 |
| Demo Q4 refiner | 2.517783 |

The shipped model leads on this excerpt. It received the full upstream training schedule. For an architecture comparison, match the train/validation documents, scored training targets, side inputs and evaluation procedure. Report parameter count, weight bytes and compute separately. The score CLI measures the neural byte head. The sample round-trip CLI below measures the full mixer's payload.

## Train on the full compressor data

Both trainers can use a completed upstream stage-(c) package. Use the directory containing `zeilen.dump`, `doc-rows.int64`, `doc-last-ziel.int32`, `ppmd-bytes.uint8`, `ppmd-probs.float16` and `z3-prior0.float16`. Upstream's package builder assembles these together in its final `pkg_check` directory. The refiner memory-maps the prior matrix, uses the upstream prompt, target and mask rules, and holds out the last 10% of documents. The full prior matrix alone occupies about 203 GB. To generate it, use the unmodified upstream commit `35d1957a` as the package builder's `ENTRY`. [Package instructions and hardware requirements](../README.md) · [Package builder](../tf/2_package/build_package.sh)

```bash
# One-time data preparation if you need to generate the package
git worktree add --detach ../cmix-stage-c 35d1957a5074db802fc134126afc77b76d931761
entry=$(cd ../cmix-stage-c && pwd)
ENTRY="$entry" ENWIK9="/absolute/path/to/enwik9" DICT="$entry/dictionary/english.dic" \
  ORDER="$entry/src/readalike_prepr/data/new_article_order" WORK="$HOME/cmix-training-data" \
  bash "$entry/training/tf/2_package/build_package.sh"
```

```bash
# From the fork root, with CUDA-enabled PyTorch and the completed package
python training/refiner/train.py --package /path/to/pkg_check --preset matched \
  --device cuda --context 512 --window 256 --batch 4 --steps 10000 --qat-steps 2500 \
  --out build-refiner/full/recursive-refiner.rrb
```

### Train the original transformer separately

The original transformer's standalone trainer and export tools are included. Prepare its Python/CUDA environment as described in the upstream training guide. Set `TFT_TREE` to the assembled upstream training tree, `PKG` to the full package above, and place the upstream beehive start weights in `$W/start/beehive.c4`. The staged script checks their SHA-256. The original training used multiple GPUs. [Environment and start weights](../README.md) · [Transformer trainer and stages](../tf/3_train/README.md)

```bash
export TFT_TREE="$HOME/cmix-transformer-tree"
export PKG="/path/to/pkg_check"
export W="$HOME/cmix-transformer-training"
cp -a training/tf/tree_upstream "$TFT_TREE"  # use a new destination
cp -a training/tf/tree_overlay/. "$TFT_TREE/"
python3 training/tf/2_package/patch_paket11.py "$TFT_TREE"
mkdir -p "$W/start"
cp /path/to/weights-poc8m_w2048_p15f_v3bf16_gestrichen_v4.c4 "$W/start/beehive.c4"
# Use the upstream training environment at $TFT_TREE/.venv, or set PY to its Python.
NGPU=8 bash training/tf/3_train/reproduce.sh
python3 training/tf/3_train/export/np2/np2_container.py np \
  "$W/t3/export/v5.c5" "$W/weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5" k fw np2 dead
```

## Run compression

For a short encode/decode comparison, run the command below yourself. It encodes the held-out sample with both models and verifies every decoded byte in a fresh process. Use `--model refiner` for just the refiner or `--sample PATH` for another excerpt in the frozen alphabet.

```bash
python3 training/refiner/benchmark.py build-refiner
```

For enwik9, build a packaged executable in a Linux working directory containing the original file named `enwik9`. Running `comp9` produces `archive9`. The total submission size is the size of both files combined.

```bash
# Start in the fork root
bash install.sh
repo=$(pwd)
mkdir -p ~/cmix-refiner-run
cd ~/cmix-refiner-run
BYTE_MODEL=refiner bash "$repo/build.sh"
./comp9
chmod +x archive9
./archive9
cmp enwik9 enwik9_uncompressed
wc -c comp9 archive9
```

This build uses the included demonstration checkpoint. To embed your own checkpoint at build time, set `REFINER_WEIGHTS` to its absolute path. Once you have a packaged Q4 refiner executable, you can install another exported checkpoint in a **new** executable. This copies the compiled code, dictionary hints and article order, and updates the embedded weights and size trailer. The checkpoint specifies the window size, block count and cycle counts at load time.

```bash
python3 "$repo/training/refiner/replace_weights.py" ./comp9 \
  "$repo/build-refiner/full/recursive-refiner.rrb" ~/cmix-next-run/comp9
```
