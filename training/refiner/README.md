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

## Full enwik9 training with bounded disk use

Use this mode for training the 301,824-parameter refiner until its validation loss stops improving. The Linux producer replays the canonical enwik9 input each epoch and sends the actual fed tokens, article resets, next-byte targets and 208 float16 PPMd probabilities to PyTorch. The trainer consumes the pipe in bounded buffers. A full pass transfers roughly 228 GB through memory. The prepared input, dictionary and native producer occupy **486 MiB on disk**. Each run also needs room for the native PPMd page file (up to 14,000 MiB apparent), preprocessing scratch and small training checkpoints. Budget about **20 GiB free** on the Linux filesystem as a planning estimate. The peak of an entire training run has not been measured. [Upstream preprocessing modes](../tf/1_dump/README.md) · [Upstream scratch usage](../../README.md#11-prerequisites)

**Prepare once** from the fork root with Ubuntu, Python 3.11+, git, tar, patch, make, clang++-18 and lld-18. The preparer checks enwik9, the generated canonical stream, the census input and the dictionary against their hashes. It builds the producer from pinned upstream code plus the row-output patch. If you have already built the upstream canonical stream, add `--reuse-bed /path/to/bed` to reuse its two verified inputs.

```bash
python3 training/refiner/live_data.py prepare /path/to/enwik9 ~/cmix-live-enwik9
```

**Train** from the fork root with CUDA-enabled PyTorch, Transformers 4.x, NumPy and filelock. The following PowerShell command uses the Windows GPU environment and starts the producer inside Ubuntu. Replace the Linux username or work path when running elsewhere.

```powershell
.\build-refiner\gpu-env\Scripts\python.exe -u training/refiner/train_live.py --dataset enwik9 --work /home/hp/cmix-live-enwik9 --wsl Ubuntu --device cuda --run-dir build-refiner/live-enwik9-training
```

On a Linux GPU host, use the Linux environment directly.

```bash
python -u training/refiner/train_live.py --dataset enwik9 --work "$HOME/cmix-live-enwik9" --device cuda --run-dir build-refiner/live-enwik9-training
```

One epoch visits the full stream once. About 5% of documents are assigned deterministically to validation. The trainer samples up to 1,024 fixed validation windows across those documents and reports their target-weighted loss in bits per byte after each full pass. Training uses batch 16, 128-token crops, a 64-token sliding window, 512 locally shuffled windows per buffer, and quantization-aware training from the first update. The best export is `best.rrb`. The default stopping rule is three consecutive passes without a cumulative improvement of 0.005 validation bits per byte over the last qualifying score. `--patience` and `--min-delta` set this rule. `--max-epochs` optionally caps a run and prints when that cap caused the stop.

Repeating the same command resumes `state.pt`. The trainer checkpoints every 32 training buffers and after every epoch. On a mid-epoch restart it regenerates PPMd history from the beginning of that pass and skips completed optimizer updates. The native producer reports progress in `producer.log`; the metadata and PPMd warm-up come before the main training rows. The extra CPU work per epoch is the cost of avoiding the dense table. Upstream's published 712-epoch experiment uses a different window, shuffling, split, optimizer and quantization schedule. [Live training code](train_live.py) · [Row producer](live_data.py) · [Original training recipe](../README.md)

### enwik8 training

The smaller benchmark uses 100,000,000 bytes of original Wikipedia XML. The preparation command accepts either uncompressed enwik8 or the original enwik9 and verifies the extracted bytes. enwik8 contains 205 distinct byte values. This mode maps them in increasing byte order to the first 205 refiner slots, preserves every input byte, and streams PPMd probabilities in that order. It holds out complete document pieces, starts a new piece at each XML page, and splits very long pieces at 131,072 bytes. The prepared data and producer use about 96 MiB. This is a raw-XML training experiment with its own checkpoint vocabulary. The enwik9 executable expects the dictionary-transformed enwik9 vocabulary. [Published enwik8 size and checksums](https://mattmahoney.net/dc/textdata.html)

Build the sample binary above from this fork and prepare enwik8 once in Linux.

```bash
python3 training/refiner/live_data.py prepare-enwik8 /path/to/enwik9 ~/cmix-live-enwik8 --binary build-refiner/cmix-refiner
```

Start or resume the separate enwik8 run from PowerShell.

```powershell
.\build-refiner\gpu-env\Scripts\python.exe -u training/refiner/train_live.py --dataset enwik8 --work /home/hp/cmix-live-enwik8 --wsl Ubuntu --device cuda --run-dir build-refiner/live-enwik8-training
```

On Linux, omit `--wsl Ubuntu` and invoke `python` from the GPU environment. Validation loss for this run is per scored raw XML byte. The enwik9 run reports per scored transformed byte.

## Compact enwik9 pilot

The compact preset keeps the high and low states, sliding attention, PPMd output gate and both 192-feature connections to cmix. It uses one shared block of width 192, three attention heads, a 192-wide SwiGLU and two rounds of one low update followed by one high update. This gives four block evaluations per byte and **301,824 trainable parameters**. The 64-token-window Q4 checkpoint is **198,016 bytes**. The Hutter score counts the weights twice because both executables carry a copy. At this window the compact weights save **9,196,160 bytes** against the matched refiner before accounting for any change in the coded payload.

The pilot uses only the original transformed enwik9 stream. The preparation command checks the upstream SHA-256, samples 16 disjoint 4,096-byte excerpts spaced across its body and holds out every fourth excerpt. Each excerpt starts with a fresh PPMd state. This produces 26 MiB of priors across both splits. The full upstream package preserves document prompts, article boundaries and prior history. The pilot measures body excerpts from a cold start. [Upstream coder-stream preparation](../tf/1_dump/README.md)

With the sample binaries built as above and PyTorch, Transformers 4.x and NumPy installed, run the following from the fork root. Use Python 3.11 or newer for preparation. Set `stream` to the upstream coder stream. The upstream preprocessing-only `cmix_stream` tool writes it as `out.cmix.temp` and exits before arithmetic coding. A local compressor run can leave the same bytes in `archive9.cmix.temp`. [Building and running the upstream stream tool](../tf/1_dump/README.md#1-the-bed-run)

```bash
stream=/absolute/path/to/archive9.cmix.temp
build=build-refiner
pilot=$build/pilot
python3 training/refiner/prepare_pilot.py "$build/cmix-refiner" "$stream" "$pilot"
python training/refiner/train.py --preset compact --demo-dir "$pilot" \
  --steps 512 --qat-steps 128 --context 128 --window 64 --batch 2 --seed 1 --threads 6 \
  --out "$build/compact/recursive-refiner.rrb" --parity "$build/compact/parity.bin"
"$build/refiner-check" "$build/compact/recursive-refiner.rrb" "$build/compact/parity.bin"
python3 training/refiner/benchmark.py "$build" --pilot "$pilot" --ppmd-baseline \
  --refiner-weights "$build/compact/recursive-refiner.rrb"
```

`--pilot` always scores predictions. It resets cmix for every held-out excerpt, then reports total bits divided by the total number of scored rows and wall time including startup. Use `--model refiner` to time only your checkpoint. Use `--excerpts` (a multiple of four) and `--excerpt-bytes` at preparation time to increase the data budget. Each additional sampled byte costs 416 bytes of PPMd priors. For a larger comparison, score both refiners on the same new split.

### Pilot results

All native scores below cover the same 16,380 next-byte predictions. Lower is better. The two 128-step runs used the same 16,382 training targets, seed, crops, 64-byte window, optimizer and final 32 steps of Q4-aware training. The 512-step compact run saw 65,517 targets sampled from the same 12 excerpts and used Q4 for its final 128 steps. Its weights are the included `recursive-refiner-compact.rrb`.

| Byte predictor | Train and export | Held-out bits per scored byte | C++ score time |
|---|---:|---:|---:|
| PPMd alone, with the refiner's probability floor | — | 3.473822 | — |
| Matched refiner, 128 steps | 154.17 s | 3.566580 | 207.67 s |
| Compact refiner, 128 steps | 20.59 s | 3.469910 | 26.33 s |
| Compact refiner, 512 steps | 39.38 s | 3.211155 | 21.00 s |
| Shipped upstream transformer | Upstream checkpoint | 1.658221 | 30.51 s |

Times are single wall-clock measurements on a Ryzen 7 8845HS laptop. PyTorch training ran on Windows with six CPU threads, including validation and parity export. Native scoring ran under WSL Ubuntu and includes four process startups and cmix predictor updates. The difference between the two compact scoring times gives an indication of timing variation for identical dimensions. The holdout applies to the two trained refiner sizes. Upstream trained on the full enwik9 stream, including these excerpts, and uses additional side inputs. The final prize score requires the actual archive and executable sizes.

For a runtime estimate, the compact C++ scorer processed 1,024-byte and 24,576-byte prefixes starting at each of the four held-out offsets. Dividing the additional 94,208 bytes by the additional 61.8 seconds gives about 1,525 body bytes/second. Applying that rate to 488 million body positions gives about **89 hours** on this laptop. Individual offsets imply roughly 49–125 hours. A full run uses the packaged binary, persistent context, preprocessing and arithmetic coding, so treat four days as a rough planning figure. The archive size and end-to-end runtime are unmeasured.

The fx2-cmix-transformer authors found that 512 tokens of attention context gave higher loss than 1,024 in their model. Beehive sped up the whole compressor by about 3% with a bit-identical fused attention kernel for its shared key/value heads. For this refiner, increasing training coverage while keeping the target window fixed is a separate experiment from shortening attention or changing the number of recursive steps. Kernel timing and prediction parity can be checked independently of either change. [fx2-cmix-transformer experiments](https://github.com/astOwOlfo/fx2-cmix-transformer-v1/blob/main/writeup.md) · [Beehive inference measurements](https://github.com/galqiwi/cmix-neif-beehive/blob/92c07c0b/README.md#32-inference-engine-srctf1)

To package the included pilot checkpoint in a fresh Linux working directory, start from the fork root, with the original `enwik9` available for your run. With an existing RRB2 compressor, the weight replacement utility at the end of this guide avoids rebuilding.

```bash
repo=$(pwd)
mkdir -p ~/cmix-refiner-compact-run
cd ~/cmix-refiner-compact-run
REFINER_WEIGHTS="$repo/recursive-refiner-compact.rrb" BYTE_MODEL=refiner bash "$repo/build.sh"
```

For the full enwik9 training recipe below, add `--preset compact` to each launch and start a new run directory. This keeps the upstream 2,048-token window and training schedule. The compact full trainer stores activations for batches of up to 8,192 padded rows and recomputes the shared blocks for larger batches to conserve memory. Set `--checkpoint-from-rows 1` to recompute on every training forward if memory is tight. Both settings produced a byte-identical exported checkpoint on the CPU test fixture. The included 512-step pilot checkpoint passed the 200-row streaming C++ versus PyTorch check with a maximum absolute probability difference of 0.000000239.

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

This build uses the included compact enwik9 pilot checkpoint. For the 9.3M demonstration checkpoint, set `REFINER_WEIGHTS="$repo/recursive-refiner.rrb"` on the build command. To embed your own checkpoint at build time, set `REFINER_WEIGHTS` to its absolute path. Once you have a packaged Q4 refiner executable, you can install another exported checkpoint in a **new** executable. This copies the compiled code, dictionary hints and article order, and updates the embedded weights and size trailer. The checkpoint specifies the window size, block count and cycle counts at load time.

```bash
python3 "$repo/training/refiner/replace_weights.py" ./comp9 \
  "$repo/build-refiner/full/recursive-refiner.rrb" ~/cmix-next-run/comp9
```
