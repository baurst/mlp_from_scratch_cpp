#pragma once
#include <string>
#include <tuple>
#include <vector>
#include "utils.h"

std::vector<std::pair<Mat2D<float>, Mat2D<float>>> read_mnist_csv(
    const std::string csv_filename, const size_t batch_size,
    const int64_t num_batches_to_load);