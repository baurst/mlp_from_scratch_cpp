#include "layer.h"
#include "utils.h"
#include <iostream>

int main() {

  auto layer = DenseLayer(3, 5);
  const std::vector<float> foo = {1.0, 2.0, 3.0};
  std::vector<std::vector<float>> inputs = {foo, foo, foo, foo};

  Mat2D<float> mat(inputs);

  std::cout << "Input: " << std::endl;
  std::cout << mat;

  const auto output = layer.call(mat);
  std::cout << "Output: " << std::endl;
  std::cout << output;
  return 0;
}