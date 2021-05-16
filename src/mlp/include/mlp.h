#pragma once
#include <memory>
#include <numeric>
#include <vector>
#include "layer.h"
#include "mlp.h"
#include "utils.h"
class MLP {
 public:
  MLP(const std::vector<size_t> layer_sizes, const size_t number_of_inputs,
      const size_t number_of_targets,
      const Initializer weight_init = RANDOM_UNIFORM,
      const Initializer bias_init = ZEROS);
  std::vector<Mat2D<float>> forward(const Mat2D<float>& input) const;
  float train(const Mat2D<float>& input, const Mat2D<float>& target,
              const Loss& loss_obj, const float learning_rate);
  Mat2D<size_t> predict(const Mat2D<float>& input) const;
  void print_debug_information(
      const std::vector<Mat2D<float>>& activations) const;

 private:
  std::vector<std::unique_ptr<Layer>> layers;
};