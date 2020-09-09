#pragma once
#include "utils.h"
#include <string>
#include <tuple>
#include <vector>

class MNISTDataset {
private:
  std::string base_dir;
  std::vector<std::pair<Mat2D<float>, Mat2D<float>>> data_samples;
  size_t m_current;
  size_t batch_size;

public:
  MNISTDataset(std::string base_dir, size_t batch_size);
  void read_mnist_csv(const std::string csv_filename);

  bool hasNext();

  float get_progress() const;

  std::pair<Mat2D<float>, Mat2D<float>> next();
};
