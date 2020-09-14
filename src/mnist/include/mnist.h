#pragma once
#include "utils.h"
#include <string>
#include <tuple>
#include <vector>

std::vector<std::pair<Mat2D<float>, Mat2D<float>>>
read_mnist_csv(const std::string csv_filename, const size_t batch_size);