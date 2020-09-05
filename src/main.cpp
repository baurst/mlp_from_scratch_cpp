#include "layer.h"
#include "utils.h"
#include <iostream>

int main() {

  auto layer = Layer(3, 4);
  std::vector<float> foo = {1.0, 2.0, 3.0};
  const auto output = layer.call(foo);

  std::cout << "Output: " << std::endl;
  print_vec(output);
  return 0;
}