#pragma once
#include "layer.h"
#include "mlp.h"
#include "utils.h"
#include <memory>
#include <numeric>
#include <vector>
class MLP {
public:
  MLP(const std::vector<size_t> layer_sizes, const size_t number_of_inputs,
      const size_t number_of_targets, const Initializer init = RANDOM_UNIFORM);
  std::vector<Mat2D<float>> forward(const Mat2D<float> &input) const;
  Mat2D<float> predict(const Mat2D<float> &input) const;
  float train(const Mat2D<float> &input, const Mat2D<float> &target,
              const L2Loss &loss_obj, const float learning_rate);

private:
  std::vector<std::unique_ptr<Layer>> layers;
};