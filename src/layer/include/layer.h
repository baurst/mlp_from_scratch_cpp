#pragma once
#include "utils.h"
#include <numeric>
#include <vector>

class Layer {
public:
  virtual Mat2D<float> forward(const Mat2D<float> &input) const = 0;
  virtual Mat2D<float> backward(const Mat2D<float> &input,
                                const Mat2D<float> &gradients_output,
                                const Mat2D<float> &learning_rate) = 0;
  virtual void print_trainable_variables() const = 0;
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
                        const Mat2D<float> &gradients_output,
                        const Mat2D<float> &learning_rate) override;
  void print_trainable_variables() const override;

private:
  Mat2D<float> weights;
  Mat2D<float> biases;
};

class RELUActivationLayer : public Layer {
public:
  RELUActivationLayer();
  ~RELUActivationLayer() override;
  Mat2D<float> forward(const Mat2D<float> &input) const override;
  Mat2D<float> backward(const Mat2D<float> &input,
                        const Mat2D<float> &gradients_output,
                        const Mat2D<float> &learning_rate) override;
  void print_trainable_variables() const override;

private:
};

class SoftmaxActivationLayer : public Layer {
public:
  SoftmaxActivationLayer();
  ~SoftmaxActivationLayer() override;
  Mat2D<float> forward(const Mat2D<float> &input) const override;
  Mat2D<float> backward(const Mat2D<float> &input,
                        const Mat2D<float> &gradients_output,
                        const Mat2D<float> &learning_rate) override;
  void print_trainable_variables() const override;

private:
};

class Loss {
public:
  virtual Mat2D<float> loss(const Mat2D<float> &predictions,
                            const Mat2D<float> &labels) const = 0;
  virtual Mat2D<float> loss_grad(const Mat2D<float> &predictions,
                                 const Mat2D<float> &labels) const = 0;
  Loss();
  ~Loss();

private:
};

class MSELoss : public Loss {
public:
  Mat2D<float> loss(const Mat2D<float> &predictions,
                    const Mat2D<float> &labels) const override;
  Mat2D<float> loss_grad(const Mat2D<float> &predictions,
                         const Mat2D<float> &labels) const override;
  MSELoss();
  ~MSELoss();

private:
};

class SoftmaxCrossEntropyWithLogitsLoss : public Loss {
public:
  Mat2D<float> loss(const Mat2D<float> &predictions,
                    const Mat2D<float> &labels) const;
  Mat2D<float> loss_grad(const Mat2D<float> &predictions,
                         const Mat2D<float> &labels) const;
  SoftmaxCrossEntropyWithLogitsLoss();
  ~SoftmaxCrossEntropyWithLogitsLoss();

private:
};

Mat2D<float> softmax(const Mat2D<float> &logits);