#include "refiner.h"

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

Model::Model(const char* path) {
  static_assert(sizeof(float) == 4, "RRB1 requires IEEE float32 on little-endian x86");
  FILE* file = std::fopen(path, "rb");
  if (!file) { std::perror(path); fail("cannot open checkpoint"); }
  char magic[4];
  uint32_t h[7];
  if (std::fread(magic, 1, 4, file) != 4 || std::memcmp(magic, "RRB1", 4) ||
      std::fread(h, sizeof(h[0]), 7, file) != 7)
    fail("expected an RRB1 checkpoint");
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
  const size_t rope_size = size_t(context_) * (width_/heads_/2);
  if (std::fseek(file, 0, SEEK_END) || std::ftell(file) != long(32 + 4*(params + 2*rope_size)) ||
      std::fseek(file, 32, SEEK_SET)) fail("checkpoint size disagrees with header");
  auto read = [&](Row& values, size_t n) {
    values.resize(n);
    if (std::fread(values.data(), sizeof(float), n, file) != n) fail("truncated checkpoint");
    for (float v : values) if (!std::isfinite(v)) fail("non-finite checkpoint value");
  };
  auto linear = [&](Linear& layer, int in, int out) {
    read(layer.weight, size_t(in)*out);
    read(layer.bias, out);
  };
  read(embeddings_, 216*d); read(high_init_, d); read(low_init_, d);
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
}

void Model::Linear::apply(const Row& input, Row& output) const {
  for (size_t i = 0; i < bias.size(); ++i)
    output[i] = bias[i] + dot(input.data(), weight.data() + i*input.size(), input.size());
}

void Model::rotate(Row& row) const {
  const int head_dim = width_ / heads_;
  const int offset = position_ * (head_dim/2);
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
  rotate(q_); rotate(k_);
  std::copy(k_.begin(), k_.end(), cache.keys.begin() + position_*width_);
  std::copy(v_.begin(), v_.end(), cache.values.begin() + position_*width_);
  const int hd = width_/heads_;
  const float scale = 1.0f/std::sqrt(float(hd));
  std::fill(attended_.begin(), attended_.end(), 0);
  for (int head = 0; head < width_; head += hd) {
    for (int t = 0; t <= position_; ++t)
      scores_[t] = dot(q_.data()+head, cache.keys.data()+t*width_+head, hd) * scale;
    softmax(scores_.data(), position_+1);
    for (int t = 0; t <= position_; ++t) {
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

void Model::step(unsigned token, float* probabilities, bool read) {
  if (token >= 216) fail("input token outside the cmix alphabet");
  // Buffers retain their addresses because the cmix bit head keeps a pointer.
  std::copy(high_init_.begin(), high_init_.end(), high_.begin());
  std::copy(low_init_.begin(), low_init_.end(), low_.begin());
  const float* embedding = embeddings_.data() + token*width_;
  int pass = 0;
  for (int h = 0; h < high_cycles_; ++h) {
    for (int i = 0; i < width_; ++i) injection_[i] = high_[i] + embedding[i];
    for (int l = 0; l < low_cycles_; ++l) update(low_, injection_, pass);
    update(high_, low_, pass);
  }
  if (read) {
    for (int v = 0; v < 208; ++v)
      probabilities[v] = dot(high_.data(), embeddings_.data()+v*width_, width_);
    softmax(probabilities, 208);
  }
  if (++position_ == context_) begin_article();
}
}  // namespace rr
