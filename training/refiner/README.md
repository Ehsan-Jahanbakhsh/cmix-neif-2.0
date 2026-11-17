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

**The only source text is enwik9.** The upstream preprocessing tools turn that file into the training package. Its largest file stores 208 PPMd probabilities per position in float16. For 488,038,367 positions this takes 203,023,960,672 bytes. The upstream builder checks the raw enwik9 SHA-256. It requires 245 GiB of working space for a full run, and upstream reports about 45 GiB peak RAM during the package checks. [Data pipeline and hardware](../README.md) · [Package builder](../tf/2_package/build_package.sh)

```bash
# Linux, from the fork root with the upstream package prerequisites installed
bash training/refiner/prepare_enwik9.sh /absolute/path/to/enwik9 /data/cmix-training
```

That command prepares the upstream package from commit `35d1957a`. If you already have its completed `pkg_check` directory, skip preparation.

### Run the refiner with the upstream training budget

On a Linux GPU host, the upstream lockfile provides the Python 3.11, CUDA PyTorch and FlashAttention 2 environment. The first training launch below uses four GPUs for b1. The second resumes from the saved epoch with eight GPUs for the remaining stages. Both commands use the same run directory. [Upstream environment](../README.md)

```bash
# Run from the fork root. Install uv first if the GPU image does not provide it.
uv sync --project training/tf/tree_upstream --python 3.11 --frozen
source training/tf/tree_upstream/.venv/bin/activate
uv pip install --python "$VIRTUAL_ENV/bin/python" 'transformers>=4.45,<5'
pkg=/data/cmix-training/pkg_check
run=/data/refiner-enwik9
torchrun --standalone --nproc-per-node=4 training/refiner/train_enwik9.py \
  --package "$pkg" --run-dir "$run" --until-stage b1
torchrun --standalone --nproc-per-node=8 training/refiner/train_enwik9.py \
  --package "$pkg" --run-dir "$run"
```

The full trainer uses all upstream training documents, the same seed-0 permutations, microbatch grouping, trailing-batch rule, prompt rows, targets, PPMd priors and loss masks. It reads the checked-in upstream stage configurations for the 2,048-position attention window, epoch counts and learning rates. Each optimizer update groups eight microbatches of up to 131,072 rows. Within a microbatch the trainer groups whole documents by length to limit GPU memory. It runs 100 + 100 + 400 BF16 epochs, 96 Q4-aware epochs and 16 FP32 cooldown epochs with the quantized weights in the forward pass. Muon updates the shared block matrices and AdamW updates the remaining parameters. Checkpoints are saved after every epoch. Repeating a launch resumes from the last saved epoch. [Refiner training code](train_enwik9.py) · [Upstream stage configurations](../tf/3_train/chain)

The refiner starts from a seeded random initialization. Upstream started from transformer weights previously trained on another preprocessing of enwik9. The refiner uses its tied byte head, PPMd gate and per-row Q4 format. The upstream model also uses predecessor and match embeddings, learned quantizer scales, integer activation quantization, Fisher allocation and a container-size penalty. These differences matter when interpreting an architecture comparison. Final evaluation scores all 487,298,491 eligible training positions and writes `$run/recursive-refiner.rrb`. Upstream reports 1.4051 in-sample bits per scored token for its strict evaluation. Archive size is measured separately with the compressor. [Upstream training results](../tf/3_train/chain/results/fp32.json)

### Shorter held-out experiment

The simpler trainer samples cropped spans and holds out the last 10% of documents. This example uses the same stage-C input files and a smaller user-selected training budget.

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
