# (MNIST) Digit Classifier using MLP from scratch

![CI status](https://github.com/baurst/mlp_from_scratch_cpp/workflows/CMake/badge.svg)

## About

If you are looking for a blazingly fast digit classifier with the most polished API and the capability to deploy to some obscure mobile device - better look elsewhere. :smile:
However, if you are trying to understand how image classification and error backpropagation work, this might be the place for you.

When working with modern Deep Learning frameworks such as [Tensorflow](https://github.com/tensorflow/tensorflow) or [PyTorch](https://github.com/pytorch/pytorch), it can sometimes be difficult to see what is actually going on under the hood.
This repository contains easily understandable implementations of a Dense Layer, LeakyReLU & Sigmoid Activation Layer, as well as Softmax-CrossEntropy Loss without using any 3rd party code. (except for the unittest framework [Catch2](https://github.com/catchorg/Catch2)).
All components, including the matrix operations are written in C++ from scratch.

## Build & Run

### Build requirements

* cmake >= 3.10
* make (or ninja)
* C++ compiler (C++17 support required)

### Build steps

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

## Performance comparison with tensorflow

As of now, the network achieves a test accuracy of 94.2% using two Dense Layers with 50 and 25 neurons each and Leaky ReLU activation (alpha=0.1), SGD with learning rate 0.01 (+decay), a batch size of 64.
An identical classifier has been implemented using tensorflow in [./plot/tf_mlp.py](https://github.com/baurst/mlp_from_scratch_cpp/plot/tf_mlp.py).

<img src="./plot/comparison.svg" width="600">

To generate this plot, just follow the instructions in  [./plot/Readme.md](https://github.com/baurst/mlp_from_scratch_cpp/plot/Readme.md).
