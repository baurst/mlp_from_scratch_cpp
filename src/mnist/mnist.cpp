#include "mnist.h"
#include "utils.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>
void MNISTDataset::read_mnist_csv(const std::string csv_filename) {
  std::ifstream ds_file(csv_filename);
  std::string line;
  const size_t img_offset = 784;
  std::vector<std::string> line_split;
  Mat2D<float> labels_one_hot(batch_size, 10, Initializer::ZEROS);
  Mat2D<float> flat_images(batch_size, 784, Initializer::ZEROS);
  size_t batch_idx = 0;
  while (std::getline(ds_file, line)) {
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
                   [](const std::string &val) { return std::stof(val); });
    for (size_t pixel_idx = 0; pixel_idx < 784; ++pixel_idx) {
      flat_images(batch_idx, pixel_idx) = image_vector[pixel_idx] / 256.0 - 0.5;
    }

    ++batch_idx;
    if (batch_idx == batch_size) {
      const auto sample = std::pair(flat_images, labels_one_hot);
      data_samples.push_back(sample);
      batch_idx = 0;
    }
  }
}

MNISTDataset::MNISTDataset(std::string base_dir, size_t batch_size)
    : base_dir(base_dir), m_current(0), batch_size(batch_size) {
  std::cout << "Parsing data from " << base_dir << "..." << std::endl;
  read_mnist_csv(base_dir);
  std::cout << "Successfully parsed " << data_samples.size() * batch_size
            << " samples from dataset" << std::endl;
}

float MNISTDataset::get_progress() const {
  return static_cast<float>(m_current) /
         static_cast<float>(data_samples.size());
}

bool MNISTDataset::hasNext() {
  bool retval = m_current < data_samples.size();
  return retval;
}
std::pair<Mat2D<float>, Mat2D<float>> MNISTDataset::next() {
  m_current += 1;
  return data_samples[m_current - 1];
}