# Recursive Refiner in cmix

The matched preset uses two shared blocks of width 512 with eight heads, a SwiGLU width of 2,298, two high cycles and three low updates per high cycle. That gives 9,284,584 trainable parameters, 1,012 fewer than the upstream transformer. Reusing the blocks gives 16 block evaluations per input token. [Original Recursive Refiner implementation](https://github.com/Ehsan-Jahanbakhsh/cramming-me/blob/15766a659deced4c18be5cc2625cbad3ebd56ae5/cramming/architectures/recursive_refiner_hf.py)

## Train a checkpoint and compile

Run these commands from the fork root in Ubuntu 24.04 on x86-64 with AVX2 and F16C. Ubuntu under WSL2 works. Use a Linux filesystem for the compression working directory. The Python model is vendored with its MIT license. C++ inference has no PyTorch dependency.

```bash
sudo apt-get update
sudo apt-get install -y clang-18 lld-18 make g++ python3-venv
python3 -m venv build-refiner/venv
source build-refiner/venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install 'transformers>=4.45,<5' numpy
python training/refiner/train.py --preset matched --steps 64 \
  --out build-refiner/recursive-refiner.rrb --parity build-refiner/parity.bin \
  --sample build-refiner/heldout.bin

make -j4 CC=clang++-18 BYTE_MODEL=refiner SAMPLE=1 cmix
cp cmix build-refiner/cmix-refiner
make -j4 CC=clang++-18 BYTE_MODEL=transformer SAMPLE=1 cmix
cp cmix build-refiner/cmix-transformer
cp weights-poc8m_w2048_tf2t3_pruned_v5k_np2.c5 build-refiner/

clang++-18 -std=c++17 -O3 -march=x86-64-v3 -ffp-model=precise -ffp-contract=off \
  training/refiner/check.cpp src/refiner/refiner.cpp src/detmath.cpp -o build-refiner/refiner-check
build-refiner/refiner-check build-refiner/recursive-refiner.rrb build-refiner/parity.bin
```

The demonstration uses upstream’s 50 KB profiling text, lowercases it, maps the frozen byte alphabet, and reserves the final 10% before drawing training windows. It uses seed 1 and a 32-position context. On the development run, 64 AdamW steps saw 4,096 training targets. Negative log likelihood fell from 12.46 to 5.46 bits per byte on a fixed sample of 512 validation targets. The 40-row C++ parity check covered prompt tokens, omitted readouts, a document reset and a context reset. Maximum probability error was 5.1 × 10⁻⁷.

## Compare compression

```bash
python3 training/refiner/benchmark.py build-refiner
```

The script encodes the same 4 KiB held-out excerpt with each full mixer and verifies every decoded byte in a fresh process. It reports payload bytes, bits per input byte, encoding time and checkpoint size. Each run gets a separate temporary working directory for cmix’s scratch files. Use `--sample PATH` for another body excerpt in the frozen byte alphabet.

The sample harness feeds body bytes directly into both engines with the real PPMd and match channels available. The enwik9 run also applies the upstream text transforms, metadata gating and prompt insertion. A prediction-only pass over the 4 KiB fixture measured **5.699 bits per scored byte for the refiner** and **2.080 for the shipped transformer**, over the same 4,095 predictions. These are byte-model log losses. The refiner had 64 demonstration training steps and the transformer had its full upstream training.

## Train on the compressor’s data

Generate the upstream stage (c) training package with a clean checkout of upstream commit `35d1957`. The refiner loader reads `zeilen.dump`, `doc-rows.int64`, `doc-last-ziel.int32` and `ppmd-bytes.uint8`. It uses upstream’s target and mask logic, scores only eligible rows, includes inserted prompt tokens as context and holds out the last 10% of documents. The large prior matrix is optional for refiner training. [Upstream training package instructions](https://github.com/marwoelk/cmix-neif-2.0/blob/35d1957a5074db802fc134126afc77b76d931761/training/README.md)

```bash
python training/refiner/train.py --package /path/to/package --preset matched \
  --device cuda --context 256 --batch 4 --steps 10000 \
  --out build-refiner/recursive-refiner.rrb
```

Use a CUDA-enabled PyTorch installation for that example. Set the context, batch size and training budget to suit the machine. For an architecture comparison, train both models on the same document split and number of scored targets, then compare held-out loss, actual payload, wall time and stored weight bytes. The shipped baseline also uses learned side-channel projections and an online fast-weight head.

## Run on enwik9

The full build embeds the checkpoint, dictionary hints and article order into `comp9`. Use the packaged executable for enwik9. The plain `make cmix` binary above is for the sample harness.

```bash
# From the fork root, prepare the Linux release toolchain and build.
bash install.sh
repo=$(pwd)
mkdir -p ~/cmix-refiner-run
cd ~/cmix-refiner-run
BYTE_MODEL=refiner REFINER_WEIGHTS="$repo/build-refiner/recursive-refiner.rrb" bash "$repo/build.sh"

# Place the original 1,000,000,000-byte file named enwik9 in this directory.
./comp9
chmod +x archive9
./archive9
cmp enwik9 enwik9_uncompressed
wc -c comp9 archive9
```

The score sums the sizes of the compressor and self-extracting archive. Each contains the weights. The matched demonstration stores 37,146,560 bytes of FP32 weights and rotary tables per copy. The shipped transformer’s quantized container is 3,744,629 bytes. This build exercises the model swap and still needs quantization and representative training to compete on total archive size. Compare with the upstream executable in a separate run directory.

## Interface to cmix

The input embedding has 216 rows for 208 byte IDs and eight prompt IDs. The output applies the first 208 rows as a tied byte head. Every high/low pass has its own causal KV history. The Python loader and C++ engine both reset at context-block boundaries, and the engine also respects upstream document resets. The first 192 high-state components feed BITLSTM and the first 192 low-state components feed the semantic cache. The refiner’s single head leaves the transformer-specific online second-head slot off during live rows. The original BITLSTM weights are retained. [Streaming implementation](../../src/refiner/refiner.cpp)
