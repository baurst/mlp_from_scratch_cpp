#pragma once
#include "utils.h"
#include <numeric>
#include <vector>

class Layer {
public:
  virtual Mat2D<float> forward(const Mat2D<float> &input) const = 0;
  virtual Mat2D<float> backward(const Mat2D<float> &input,
                                const Mat2D<float> &gradients_output) = 0;
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
  Mat2D<float> backward(const Mat2D<float> &input,
                        const Mat2D<float> &gradients_output) override;

private:
  Mat2D<float> weights;
  Mat2D<float> biases;
};

class ActivationLayer : public Layer {
public:
  ActivationLayer();
  ~ActivationLayer() override;
  Mat2D<float> forward(const Mat2D<float> &input) const override;
  Mat2D<float> backward(const Mat2D<float> &input,
                        const Mat2D<float> &gradients_output) override;

private:
};

class L2Loss {
public:
  Mat2D<float> loss(const Mat2D<float> &predictions,
                    const Mat2D<float> &labels) const;
  Mat2D<float> loss_grad(const Mat2D<float> &predictions,
                         const Mat2D<float> &labels) const;
  L2Loss();
  ~L2Loss();

private:
};