#include "neuron.h"
#include "utils.h"
#include <iostream>
Neuron::Neuron(size_t number_of_inputs) {

  weights.resize(number_of_inputs);
  bias = 0.0;
  std::cout << "Creating neuron with " << weights.size() << " weights"
            << std::endl;
}

std::vector<float>
Neuron::call(const std::vector<std::vector<float>> input) const {

  std::vector<float> output;
  const size_t batch_size = input.size();

  output.resize(batch_size);
  for (size_t batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    const std::vector<float> batch_el = input[batch_idx];
    const float dot_prod = bias + dot(weights, batch_el);
    // std::transform(dot_prod.begin(), dot_prod.end(), dot_prod.begin(),
    //                [](float el) { return el + bias; });
    output[batch_idx] = dot_prod;
  }
  return output;
}