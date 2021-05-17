# (MNIST) Digit Classifier using MLP from scratch

![CI status](https://github.com/baurst/mlp_from_scratch_cpp/workflows/CMake/badge.svg)

## About

If you are looking for a blazingly fast digit classifier with the most polished API and the capability to deploy to some obscure mobile device... you better look elsewhere. :smile:
However, if you are interested in understanding how image classification and error backpropagation work, this might be the place for you.

When working with modern Deep Learning frameworks such as [Tensorflow](https://github.com/tensorflow/tensorflow) or [PyTorch](https://github.com/pytorch/pytorch), it can sometimes be difficult to see what is actually going on under the hood.
This repository contains easily understandable implementations of a Dense Layer, LeakyReLU & Sigmoid Activation Layer, as well as Softmax-CrossEntropy Loss without using any 3rd party code. (except for the unittest framework [Catch2](https://github.com/catchorg/Catch2)).
All components, including the matrix operations are written in C++ from scratch.

## Build & Run

### Build requirements

* cmake >= 3.10
* make (or ninja)
* C++ compiler (C++17 support required)

### Steps to build

```bash
git clone --recursive https://github.com/baurst/mlp_from_scratch_cpp.git
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release  # for debug build use -DCMAKE_BUILD_TYPE=Debug
make -j4

# to run the tests
./test/tests
```

### Steps to run

```bash
# in build directory, download MNIST dataset as .csv
wget https://pjreddie.com/media/files/mnist_train.csv
wget https://pjreddie.com/media/files/mnist_test.csv

# train the classifier by providing absolute paths to datasets as arguments
./src/main mnist_train.csv mnist_test.csv
```

## Explanation

For each train step, the train method of the MLP is called with a mini-batch of images and corresponding labels, called input & target.

```C++
float MLP::train(const Mat2D<float>& input, const Mat2D<float>& target,
                 const Loss& loss_obj, const float learning_rate) {
```

First, the forward pass is performed, i.e. the activations of each layer for the mini-batch "input" are stored like so:

```C++
  const auto activations = this->forward(input);
  const auto logits = activations.back();
```

In the backward pass, we propagate the gradient of the loss w.r.t. to its input back through the network to update the weights.
As the name suggests, backpropagation starts at the back of the computation graph, in our case from the loss layer.
During the backward pass, each layer updates its trainable variables with the gradient (scaled by the learning rate) and returns the gradient of the loss w.r.t. its inputs.

```C++
  auto grad = loss_obj.loss_grad(logits, target_label);

  for (int32_t layer_idx = this->layers.size() - 1; layer_idx >= 0;
       --layer_idx) {
    const auto layer_input = activations[layer_idx];
    grad = this->layers[layer_idx]->backward(layer_input, grad, learning_rate);
  }
```

(For details on how each layer implements the backward method, see [./src/layer/layer.cpp](https://github.com/baurst/mlp_from_scratch_cpp/blob/master/src/layer/layer.cpp).)
The weight update for this mini-batch is now complete.
Finally, the loss is returned for logging purposes.

```C++
  const auto loss = loss_obj.loss(logits, target_label);
  const auto avg_loss = loss.reduce_mean();
  return avg_loss;
}
```

## Performance comparison with tensorflow

As of now, the network achieves a test accuracy of 94.2% using two Dense Layers with 50 and 25 neurons each and Leaky ReLU activation (alpha=0.1), SGD with learning rate 0.01 (with decay) and a batch size of 64.
An identical classifier has been implemented using tensorflow in [./plot/tf_mlp.py](https://github.com/baurst/mlp_from_scratch_cpp/blob/master/plot/tf_mlp.py).

<img src="./plot/comparison.svg" width="600">

The code to generate this plot is also included in this repo, just follow the instructions in  [./plot/Readme.md](https://github.com/baurst/mlp_from_scratch_cpp/blob/master/plot/Readme.md).
