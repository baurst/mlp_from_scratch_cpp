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
  Mat2D<float> call(const Mat2D<float> &input) const;

private:
  std::vector<std::unique_ptr<Layer>> layers;
  size_t number_of_inputs;
  size_t number_of_targets;
};