#include "layer.h"
#include "utils.h"
#include <iostream>

int main() {

  auto layer = Layer(3, 5);
  const std::vector<float> foo = {1.0, 2.0, 3.0};
  std::vector<std::vector<float>> inputs = {foo, foo};

  std::cout << "Input: " << std::endl;
  print_vec_of_vecs(inputs);

  const auto output = layer.call(inputs);

  std::cout << "Output: " << std::endl;
  print_vec_of_vecs(output);
  return 0;
}