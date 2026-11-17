#include "refiner.h"
#include "../tf1/opt/glue.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#ifdef __AVX2__
#include <immintrin.h>
#endif

namespace rr {
namespace {

[[noreturn]] void fail(const char* message) {
  std::fprintf(stderr, "Recursive Refiner: %s\n", message);
  std::exit(1);
}

float dot(const float* a, const float* b, int n) {
  float sum = 0;
  int i = 0;
#ifdef __AVX2__
  __m256 acc = _mm256_setzero_ps();
  for (; i + 8 <= n; i += 8)
    acc = _mm256_add_ps(acc, _mm256_mul_ps(_mm256_loadu_ps(a+i), _mm256_loadu_ps(b+i)));
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  for (float value : lanes) sum += value;
#endif
  for (; i < n; ++i) sum += a[i] * b[i];
  return sum;
}

void rms(const std::vector<float>& x, const std::vector<float>& weight, std::vector<float>& out) {
  const float scale = 1.0f / std::sqrt(dot(x.data(), x.data(), x.size()) / x.size() + 1e-5f);
  for (size_t i = 0; i < x.size(); ++i) out[i] = (x[i] * scale) * weight[i];
}

void softmax(float* values, int n) {
  const float top = *std::max_element(values, values+n);
  float sum = 0;
  for (int i = 0; i < n; ++i) sum += (values[i] = std::exp(values[i] - top));
  for (int i = 0; i < n; ++i) values[i] /= sum;
}
}  // namespace

float Model::Matrix::product(const Row& x, int row) const {
  const size_t n = x.size(), offset = size_t(row)*n;
  if (codes.empty()) return dot(x.data(), fp32.data()+offset, n);
  const int8_t* w = codes.data()+offset;
  size_t i = 0;
  float sum = 0;
#ifdef __AVX2__
  __m256 acc = _mm256_setzero_ps();
  for (; i+8 <= n; i += 8) {
    __m128i bytes = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(w+i));
    __m256 weight = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bytes));
    acc = _mm256_add_ps(acc, _mm256_mul_ps(weight, _mm256_loadu_ps(x.data()+i)));
  }
  alignas(32) float lanes[8];
  _mm256_store_ps(lanes, acc);
  for (float v : lanes) sum += v;
#endif
  for (; i < n; ++i) sum += w[i]*x[i];
  return sum * scales[row];
}

Model::Model(const char* path) {
  static_assert(sizeof(float) == 4, "RRB1 requires IEEE float32 on little-endian x86");
  FILE* file = std::fopen(path, "rb");
  if (!file) { std::perror(path); fail("cannot open checkpoint"); }
  char magic[4];
  uint32_t h[7];
  if (std::fread(magic, 1, 4, file) != 4 ||
      (std::memcmp(magic, "RRB1", 4) && std::memcmp(magic, "RRB2", 4)) ||
      std::fread(h, sizeof(h[0]), 7, file) != 7)
    fail("expected an RRB1 or RRB2 checkpoint");
  rolling_ = magic[3] == '2';
  // width, heads, FFN width, shared layers, H cycles, L cycles, context
  if (h[0] < 192 || h[0] > 1024 || !h[1] || h[0] % h[1] || (h[0]/h[1]) % 2 ||
      !h[2] || h[2] > 8192 || !h[3] || h[3] > 32 || !h[4] || h[4] > 8 ||
      !h[5] || h[5] > 8 || !h[6] || h[6] > 2048)
    fail("unsupported model dimensions");
  width_ = h[0]; heads_ = h[1]; inner_ = h[2];
  high_cycles_ = h[4]; low_cycles_ = h[5]; context_ = h[6];
  const size_t passes = size_t(h[3]) * high_cycles_ * (1 + low_cycles_);
  const size_t cache_size = size_t(width_) * context_;
  if (2 * passes * cache_size > (1u << 27)) fail("KV cache exceeds 512 MiB");
  const size_t d = width_, f = inner_;
  const size_t params = 218*d + h[3]*(4*d*d + 7*d + (3*d+2)*f);
  const size_t rope_size = size_t(context_) * (rolling_ ? 2 : 1) * (width_/heads_/2);
  auto packed_bytes = [](size_t rows, size_t cols) { return rows*(4+(cols+1)/2); };
  size_t bytes = 32 + 4*(params + 2*rope_size);
  if (rolling_) {
    size_t block_bytes = 4*(7*d+2*f) + 4*packed_bytes(d,d)
                       + packed_bytes(2*f,d) + packed_bytes(d,f);
    bytes = 32 + 4*(3*d + 2*rope_size) + packed_bytes(216,d) + h[3]*block_bytes;
  }
  if (std::fseek(file, 0, SEEK_END) || std::ftell(file) != long(bytes) ||
      std::fseek(file, 32, SEEK_SET)) fail("checkpoint size disagrees with header");
  auto read = [&](Row& values, size_t n) {
    values.resize(n);
    if (std::fread(values.data(), sizeof(float), n, file) != n) fail("truncated checkpoint");
    for (float v : values) if (!std::isfinite(v)) fail("non-finite checkpoint value");
  };
  auto matrix = [&](Matrix& m, size_t rows, size_t cols) {
    if (!rolling_) { read(m.fp32, rows*cols); return; }
    read(m.scales, rows);
    for (float scale : m.scales) if (!(scale > 0)) fail("invalid Q4 scale");
    m.codes.resize(rows*cols);
    std::vector<uint8_t> packed((cols+1)/2);
    for (size_t r = 0; r < rows; ++r) {
      if (std::fread(packed.data(), 1, packed.size(), file) != packed.size()) fail("truncated Q4 row");
      for (size_t i = 0; i < cols; ++i) {
        int nibble = (packed[i/2] >> (4*(i%2))) & 15;
        if (nibble == 15) fail("invalid Q4 code");
        m.codes[r*cols+i] = int8_t(nibble-7);
      }
    }
  };
  auto linear = [&](Linear& layer, int in, int out) {
    matrix(layer.weight, out, in);
    read(layer.bias, out);
  };
  matrix(embeddings_, 216, d); read(high_init_, d); read(low_init_, d);
  if (rolling_) read(prior_gate_, d);
  read(cos_, rope_size); read(sin_, rope_size);
  blocks_.resize(h[3]);
  for (Block& b : blocks_) {
    read(b.norm1, d); read(b.norm2, d);
    for (Linear* proj : {&b.q, &b.k, &b.v, &b.out}) linear(*proj, d, d);
    linear(b.fc1, d, 2*f); linear(b.fc2, f, d);
  }
  std::fclose(file);
  caches_.resize(passes);
  for (Cache& c : caches_) { c.keys.resize(cache_size); c.values.resize(cache_size); }
  for (Row* r : {&high_, &low_, &injection_, &norm_, &q_, &k_, &v_, &attended_, &projected_})
    r->resize(d);
  gates_.resize(2*f); ff_.resize(f); scores_.resize(context_);
  embedding_.resize(d); previous_q_.resize(d);
}

void Model::Linear::apply(const Row& input, Row& output) const {
  for (size_t i = 0; i < bias.size(); ++i)
    output[i] = bias[i] + weight.product(input, i);
}

void Model::rotate(Row& row, int phase) const {
  const int head_dim = width_ / heads_;
  const int offset = phase * (head_dim/2);
  for (int head = 0; head < width_; head += head_dim) {
    for (int i = 0; i < head_dim; i += 2) {
      float c = cos_[offset+i/2], s = sin_[offset+i/2];
      float a = row[head+i], b = row[head+i+1];
      row[head+i] = a*c - b*s;
      row[head+i+1] = a*s + b*c;
    }
  }
}

void Model::attend(const Block& b, Cache& cache, Row& state) {
  rms(state, b.norm1, norm_);
  b.q.apply(norm_, q_); b.k.apply(norm_, k_); b.v.apply(norm_, v_);
  // Keys use their physical slot's rotary phase. For keys from the previous
  // lap, rotating the query by one extra window gives the exact relative age.
  if (history_ > position_) {
    std::copy(q_.begin(), q_.end(), previous_q_.begin());
    rotate(previous_q_, position_+context_);
  }
  rotate(q_, position_); rotate(k_, position_);
  std::copy(k_.begin(), k_.end(), cache.keys.begin() + position_*width_);
  std::copy(v_.begin(), v_.end(), cache.values.begin() + position_*width_);
  const int hd = width_/heads_;
  const float scale = 1.0f/std::sqrt(float(hd));
  std::fill(attended_.begin(), attended_.end(), 0);
  for (int head = 0; head < width_; head += hd) {
    int n = std::min(history_+1, context_);
    for (int t = 0; t < n; ++t) {
      const Row& query = t <= position_ ? q_ : previous_q_;
      scores_[t] = dot(query.data()+head, cache.keys.data()+t*width_+head, hd) * scale;
    }
    softmax(scores_.data(), n);
    for (int t = 0; t < n; ++t) {
      const float* values = cache.values.data()+t*width_+head;
      for (int i = 0; i < hd; ++i) attended_[head+i] += scores_[t]*values[i];
    }
  }
  b.out.apply(attended_, projected_);
  for (int i = 0; i < width_; ++i) state[i] += projected_[i];
}

void Model::update(Row& state, const Row& injection, int& pass) {
  for (int i = 0; i < width_; ++i) state[i] += injection[i];
  for (const Block& b : blocks_) {
    attend(b, caches_[pass++], state);
    rms(state, b.norm2, norm_);
    b.fc1.apply(norm_, gates_);
    for (int i = 0; i < inner_; ++i)
      ff_[i] = (gates_[i] / (1.0f+std::exp(-gates_[i]))) * gates_[inner_+i];
    b.fc2.apply(ff_, projected_);
    for (int i = 0; i < width_; ++i) state[i] += projected_[i];
  }
}

void Model::step(unsigned token, float* probabilities, bool read, const float* prior) {
  if (token >= 216) fail("input token outside the cmix alphabet");
  // Buffers retain their addresses because the cmix bit head keeps a pointer.
  std::copy(high_init_.begin(), high_init_.end(), high_.begin());
  std::copy(low_init_.begin(), low_init_.end(), low_.begin());
  for (int i = 0; i < width_; ++i) {
    size_t j = size_t(token)*width_+i;
    embedding_[i] = rolling_ ? embeddings_.codes[j]*embeddings_.scales[token] : embeddings_.fp32[j];
  }
  int pass = 0;
  for (int h = 0; h < high_cycles_; ++h) {
    for (int i = 0; i < width_; ++i) injection_[i] = high_[i] + embedding_[i];
    for (int l = 0; l < low_cycles_; ++l) update(low_, injection_, pass);
    update(high_, low_, pass);
  }
  if (read) {
    for (int v = 0; v < 208; ++v)
      probabilities[v] = embeddings_.product(high_, v);
    if (rolling_) {
      if (!prior) fail("RRB2 readout requires a cmix prior");
      float inv_rms = 1.0f/std::sqrt(dot(high_.data(), high_.data(), width_)/width_ + 1e-5f);
      float gate = dot(high_.data(), prior_gate_.data(), width_)*inv_rms;
      fx2::opt::head_plm_softcap_softmax(probabilities, prior, gate, probabilities);
    } else softmax(probabilities, 208);
  }
  history_ = std::min(history_+1, context_);
  if (++position_ == context_) {
    position_ = 0;
    if (!rolling_) history_ = 0;
  }
}
}  // namespace rr
