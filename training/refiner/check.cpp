// Compare the streaming implementation to full-sequence PyTorch predictions.
#include "../../src/refiner/refiner.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: refiner-check MODEL.rrb parity.bin\n");
    return 2;
  }
  rr::Model model(argv[1]);
  FILE* file = std::fopen(argv[2], "rb");
  if (!file) { std::perror(argv[2]); return 1; }
  char header[4];
  bool priors = std::fread(header, 1, 4, file) == 4 && std::memcmp(header, "RRP2", 4) == 0;
  if (!priors) std::rewind(file);
  if (model.uses_prior() != priors) { std::fprintf(stderr, "wrong parity fixture format\n"); return 1; }
  float max_prob = 0, max_state = 0;
  int rows = 0;
  for (int token; (token = std::fgetc(file)) != EOF; ++rows) {
    int reset = std::fgetc(file), read = std::fgetc(file);
    float expected[208 + 2*192], prob[208], prior[208];
    if (reset < 0 || reset > 1 || read < 0 || read > 1 ||
        (priors && std::fread(prior, sizeof(float), 208, file) != 208) ||
        std::fread(expected, sizeof(float), 592, file) != 592) return 1;
    if (reset) model.begin_article();
    model.step(token, prob, read, priors ? prior : nullptr);
    for (int i = read ? 0 : 208; i < 592; ++i) {
      float actual = i < 208 ? prob[i] :
          i < 400 ? model.last_hidden()[i-208] : model.last_key()[i-400];
      float error = std::abs(actual - expected[i]);
      float& maximum = i < 208 ? max_prob : max_state;
      maximum = std::max(maximum, error);
      float tolerance = i < 208 ? 1e-4f : 0.001f + 1e-4f*std::abs(expected[i]);
      if (!(error <= tolerance)) {
        std::fprintf(stderr, "row %d column %d expected %.8g got %.8g\n",
                     rows, i, expected[i], actual);
        return 1;
      }
    }
  }
  const bool ok = rows > 0 && !std::ferror(file);
  std::fclose(file);
  std::printf("%d rows, max absolute error: probabilities %.8g, states %.8g\n",
              rows, max_prob, max_state);
  return ok ? 0 : 1;
}
