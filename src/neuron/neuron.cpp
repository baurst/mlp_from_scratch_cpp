#include "neuron.h"

Neuron::Neuron(size_t number_of_inputs, size_t number_of_outputs) {
    weights.resize(number_of_inputs);
    bias = 0.0;
}

float Neuron::call(const std::vector<float> input ){
    return bias + dot(weights, input);
}