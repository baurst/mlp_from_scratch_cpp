#include "mlp.h"
#include "layer.h"
#include "utils.h"
#include <iostream>
#include <math.h>
#include <memory>
#include <stdexcept>
#include <vector>

MLP::MLP(const std::vector<size_t> layer_sizes, const size_t number_of_inputs,
         const size_t number_of_targets, const Initializer weight_init,
         const Initializer bias_init)
{
  size_t input_size = number_of_inputs;
  size_t layer_idx = 0;
  for (const size_t layer_size : layer_sizes)
  {
    std::cout << "Layer " << layer_idx << ": ";
    layers.push_back(std::make_unique<DenseLayer>(input_size, layer_size,
                                                  weight_init, bias_init));
    input_size = layer_size; // for the next layer
    layer_idx++;
    std::cout << "Layer " << layer_idx << ": ";
    layers.push_back(std::make_unique<LeakyRELUActivationLayer>(0.1));
  }
  std::cout << "Layer " << layer_idx << ": ";
  layers.push_back(std::make_unique<DenseLayer>(input_size, number_of_targets));
  layer_idx++;
}

std::vector<Mat2D<float>> MLP::forward(const Mat2D<float> &input) const
{
  std::vector<Mat2D<float>> activations;
  activations.reserve(this->layers.size() + 1);
  activations.push_back(input);
  size_t layer_idx = 0;
  for (const auto &layer : layers)
  {
    const auto tmp_out = layer->forward(activations.back());
    activations.push_back(tmp_out);

    layer_idx++;
  }
  return activations;
}

float MLP::train(const Mat2D<float> &input, const Mat2D<float> &target_label,
                 const Loss &loss_obj, const float learning_rate)
{
  const auto activations = this->forward(input);
  const auto logits = activations.back();
  const auto loss = loss_obj.loss(logits, target_label);

  auto grad = loss_obj.loss_grad(logits, target_label);
  if (std::isnan(grad.reduce_mean()))
  {
    this->print_debug_information(activations);
    std::cout.flush();
    throw std::runtime_error("Encountered NAN in Gradient, we are doomed! "
                             "Maybe try lowering the learning rate.");
  }
  const auto lr_mat = Mat2D<float>(
      1, 1, {learning_rate});

  for (int32_t layer_idx = this->layers.size() - 1; layer_idx >= 0;
       --layer_idx)
  {
    const auto layer_input = activations[layer_idx];

    grad = this->layers[layer_idx]->backward(layer_input, grad, lr_mat);
  }
  const auto avg_loss = loss.reduce_mean();

  if (std::isnan(avg_loss))
  {
    this->print_debug_information(activations);
    std::cout.flush();
    throw std::runtime_error(
        "Encountered NAN in loss! Maybe try lowering the learning rate.");
  }
  return avg_loss;
}

Mat2D<size_t> MLP::predict(const Mat2D<float> &input) const
{
  const auto activations = this->forward(input);
  const auto logits = activations.back();
  const auto argmax_indices = logits.argmax(1);
  return argmax_indices;
}

void MLP::print_debug_information(
    const std::vector<Mat2D<float>> &activations) const
{
  for (size_t layer_idx = 0; layer_idx < this->layers.size(); ++layer_idx)
  {
    std::cout << "Layer: " << layer_idx << " activation:" << std::endl;
    std::cout << activations[layer_idx] << std::endl;
    std::cout << "Layer: " << layer_idx << " trainable variables:" << std::endl;
    this->layers[layer_idx]->print_trainable_variables();
  }
}