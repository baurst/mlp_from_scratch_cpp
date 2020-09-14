#pragma once
#include "utils.h"
#include <numeric>
#include <vector>

class Layer {
public:
  virtual Mat2D<float> forward(const Mat2D<float> &input) const = 0;
  Layer();
  virtual ~Layer() = 0;

private:
};

class DenseLayer : public Layer {
public:
  DenseLayer(size_t number_of_inputs, size_t number_of_neurons,
             Initializer init = RANDOM_UNIFORM);
  ~DenseLayer() override;
  Mat2D<float> forward(const Mat2D<float> &input) const override;

private:
  Mat2D<float> weights;
  Mat2D<float> biases;
};

class ActivationLayer : public Layer {
public:
  ActivationLayer();
  ~ActivationLayer() override;
  Mat2D<float> forward(const Mat2D<float> &input) const override;

private:
};