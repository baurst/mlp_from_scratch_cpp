# MNIST Digit Classifier using MLP from scratch

![CI status](https://github.com/baurst/mlp_from_scratch_cpp/workflows/CMake/badge.svg)

If you are looking for a blazingly fast digit classifier with the most polished API - look elsewhere.
But if you are trying to understand how image classification and backpropagation really work, this might be the place for you.

When working with modern Deep Learning frameworks such as [Tensorflow](https://github.com/tensorflow/tensorflow) or [PyTorch](https://github.com/pytorch/pytorch), it is difficult to see through all the generated code what is actually going on under the hood.
This repository contains easily understandable implementations of a Dense Layer, LeakyReLU & Sigmoid Activation Layer, as well as Softmax-CrossEntropy Loss without using any 3rd party code. (except for the unittest framework [Catch2](https://github.com/catchorg/Catch2)).
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

## Performance

As it stands, the network achieves a Test Accuracy of 94.6%,using two Dense Layers with 50 and 25 neurons each.

To generate this fancy plot, just:

```bash
# run experiment, redirect output to log files
./src/main mnist_train.csv mnist_test.csv >> /tmp/mlp_log.txt
# use gnuplot
cat /tmp/mlp_log.txt | grep 'Online VAL' | awk '{ print $2" "$9 }' | gnuplot -p -e "set key right center; set title \"Online Validation\"; set xlabel \"Train Iterations\"; plot '-' using 1:2 with lines title \"Accuracy\" linetype 7 lw 4"
```

## TODO

* add loss curve using grep, awk, gnuplot: loss, accuracy
* tensorflow baseline:
** lr decay
** val dataset
** softmax crossentropy with logits
