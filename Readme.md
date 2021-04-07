# MNIST Digit Classifier using MLP from scratch

![CI status](https://github.com/baurst/mlp_from_scratch_cpp/workflows/CMake/badge.svg)

If you are looking for a blazingly fast digit classifier with the most polished API - look elsewhere.
But if you are trying to understand how image classification and backpropagation really work, this might be the place for you.

When working with modern Deep Learning frameworks like Tensorflow or PyTorch, it is difficult to see through all the generated code what is actually going on under the hood.
This repository contains easily understandable implementations of a Dense Layer, LeakyReLU & Sigmoid Activation Layer, as well as Softmax-CrossEntropy Loss without using any 3rd party code (except for the test framework).
All components, including the matrix operations are written in C++ from scratch.

## Build & Run

```bash
git clone --recursive https://github.com/baurst/mlp_from_scratch_cpp.git
mkdir build
cd build

# Download MNIST dataset as .csv
wget https://pjreddie.com/media/files/mnist_train.csv
wget https://pjreddie.com/media/files/mnist_test.csv

# train the classifier by providing path to datasets as arguments
./src/main mnist_train.csv mnist_test.csv

# to run the tests
./test/tests

```
