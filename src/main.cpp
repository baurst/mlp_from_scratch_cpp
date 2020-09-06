#include "layer.h"
#include "mlp.h"
#include "utils.h"
#include <iostream>

int main() {

  std::vector<size_t> layer_sizes = {5, 8};

  auto mlp = MLP(layer_sizes, 3, 2);
  const std::vector<float> foo = {1.0, -2.0, 3.0};
  std::vector<std::vector<float>> inputs = {foo, foo, foo, foo};

  Mat2D<float> mat(inputs);

  std::cout << "Input: " << std::endl;
  std::cout << mat;

  const auto output = mlp.call(mat);
  std::cout << "Output: " << std::endl;
  std::cout << output;
  return 0;
}