#pragma once

#include <cstdio>
#include <cstdint>
#include <vector>

namespace rr {

// Streaming equivalent of the nested, pre-norm causal RecursiveRefinerModel.
// Weights are shared across cycles. Each invocation needs its own KV history.
class Model {
 public:
  explicit Model(const char* path);
  void begin_article() { position_ = 0; history_ = 0; }
  void step(unsigned token, float* probabilities, bool read = true, const float* prior = nullptr);
  bool uses_prior() const { return rolling_; }
  bool needs_prior_on_unread_rows() const { return score_feedback_; }
  const float* last_hidden() const { return high_.data(); }
  const float* last_key() const { return low_.data(); }
  static constexpr int hidden_size() { return 192; }  // cmix feature width

 private:
  using Row = std::vector<float>;
  struct Matrix {
    Row fp32, scales;
    std::vector<int8_t> codes;
    float product(const Row& x, int row) const;
  };
  struct Linear {
    Matrix weight;
    Row bias;
    void apply(const Row& input, Row& output) const;
  };
  struct Block {
    Row norm1, norm2;
    Linear q, k, v, out, fc1, fc2;
  };
  struct Cache { Row keys, values; };
  void update(Row& state, const Row& injection, int& pass);
  void revise(Row& state, const Row& injection, int& pass, bool project_scores = true);
  void attend(const Block& block, Cache& cache, Row& state);
  void rotate(Row& row, int phase) const;

  int width_, heads_, inner_, high_cycles_, low_cycles_, context_;
  int position_ = 0, history_ = 0;
  bool rolling_ = false;
  bool score_feedback_ = false;
  Matrix embeddings_;
  Row high_init_, low_init_, cos_, sin_, prior_gate_, embedding_, previous_q_;
  std::vector<Block> blocks_;
  std::vector<Cache> caches_;
  Row high_, low_, injection_, norm_, q_, k_, v_, attended_, projected_, gates_, ff_, scores_;
  Row byte_embeddings_, score_input_, base_scores_, byte_scores_, byte_probs_;
};

}  // namespace rr
