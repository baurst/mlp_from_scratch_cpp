#include "mlp.h"
#include "layer.h"
#include "utils.h"
#include <iostream>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <vector>

MLP::MLP(const std::vector<size_t> layer_sizes, const size_t number_of_inputs,
         const size_t number_of_targets, const Initializer init) {
  size_t input_size = number_of_inputs;

  for (const size_t layer_size : layer_sizes) {
    layers.push_back(
        std::make_unique<DenseLayer>(input_size, layer_size, init));
    input_size = layer_size;
    layers.push_back(std::make_unique<ActivationLayer>());
  }
  layers.push_back(std::make_unique<DenseLayer>(input_size, number_of_targets));
}

Mat2D<float> MLP::predict(const Mat2D<float> &input) const {
  auto tmp_in = input;
  for (const auto &layer : layers) {
    const auto tmp_out = layer->forward(tmp_in);
    tmp_in = tmp_out;
  }
  Mat2D<float> predictions(input.get_num_rows(), 1, Initializer::ZEROS);

  for (size_t row_idx = 0; row_idx < input.get_num_rows(); ++row_idx) {
    size_t max_idx = 0;
    float max_val = std::numeric_limits<float>::lowest();
    for (size_t col_idx = 0; col_idx < tmp_in.get_num_cols(); ++col_idx) {
      if (tmp_in(row_idx, col_idx) > max_val) {
        max_val = tmp_in(row_idx, col_idx);
        max_idx = col_idx;
      }
    }
    predictions(row_idx, 1) = static_cast<float>(max_idx);
  }
  return predictions;
}

std::vector<Mat2D<float>> MLP::forward(const Mat2D<float> &input) const {
  std::vector<Mat2D<float>> activations;
  activations.reserve(this->layers.size() + 1);
  activations.push_back(input);
  auto tmp_in = input;
  for (const auto &layer : layers) {
    const auto tmp_out = layer->forward(tmp_in);
    activations.push_back(tmp_out);
    tmp_in = tmp_out;
  }
  return activations;
}

float MLP::train(const Mat2D<float> &input, const Mat2D<float> &target_label,
                 const MSELoss &loss_obj, const Mat2D<float> &learning_rate) {
  const auto activations = this->forward(input);
  const auto logits = activations.back();

  const auto loss = loss_obj.loss(logits, target_label);
  auto error =
      (loss_obj.loss_grad(logits, target_label)).hadamard_product(logits);

  for (int32_t layer_idx = this->layers.size() - 1; layer_idx >= 0;
       --layer_idx) {
    error = this->layers[layer_idx]->backward(activations[layer_idx], error,
                                              learning_rate);
  }

  return loss.reduce_mean();
}