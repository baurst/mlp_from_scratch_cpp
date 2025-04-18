#include "mnist.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <tuple>
#include <vector>

#include "utils.h"
std::vector<std::pair<Mat2D<float>, Mat2D<float>>> read_mnist_csv(
    const std::string csv_filename, const size_t batch_size,
    const int64_t num_batches_to_load) {
  std::cout << "Loading MNIST dataset from " << csv_filename << std::endl;
  std::vector<std::pair<Mat2D<float>, Mat2D<float>>> dataset;
  std::ifstream ds_file(csv_filename);
  std::string line;
  const size_t img_offset = 784;
  size_t batch_idx = 0;
  size_t num_batches_loaded = 0;
  Mat2D<float> flat_images(batch_size, 784, Initializer::ZEROS);
  Mat2D<float> labels_one_hot(batch_size, 10, Initializer::ZEROS);
  while (std::getline(ds_file, line)) {
    std::vector<std::string> line_split;
    std::stringstream ss(line);

    while (ss.good()) {
      std::string substr;
      std::getline(ss, substr, ',');
      line_split.push_back(substr);
    }
    const auto label = static_cast<size_t>(std::stoul(line_split[0]));
    labels_one_hot(batch_idx, label) = 1.0;
    std::vector<std::string>(line_split.begin() + 1,
                             line_split.begin() + img_offset)
        .swap(line_split);

    std::vector<float> image_vector(line_split.size());
    std::transform(line_split.begin(), line_split.end(), image_vector.begin(),
                   [](const std::string& val) { return std::stof(val); });
    for (size_t pixel_idx = 0; pixel_idx < 784; ++pixel_idx) {
      flat_images(batch_idx, pixel_idx) = image_vector[pixel_idx] / 256.0 - 0.5;
    }

    ++batch_idx;
    if (batch_idx == batch_size) {
      dataset.push_back(std::make_pair(flat_images, labels_one_hot));
      flat_images = Mat2D<float>(batch_size, 784, Initializer::ZEROS);
      labels_one_hot = Mat2D<float>(batch_size, 10, Initializer::ZEROS);
      batch_idx = 0;
      num_batches_loaded++;
      if (num_batches_to_load > 0 &&
          static_cast<int64_t>(num_batches_loaded) == num_batches_to_load) {
        break;
      }
    }
  }
  std::cout << "Loaded " << dataset.size() << " batches of " << batch_size
            << " samples. Dropped remainder: " << batch_idx << std::endl;

  std::random_device random_device;
  std::mt19937 gen(random_device());

  std::shuffle(dataset.begin(), dataset.end(), gen);
  return dataset;
}