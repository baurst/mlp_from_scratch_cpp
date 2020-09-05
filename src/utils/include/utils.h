#pragma once
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
template <typename T> void print_vec(std::vector<T> const &vec) {
  std::cout << "[";
  for (const auto el : vec) {
    std::cout << el << ',';
  }
  std::cout << "]" << std::endl;
}

template <typename T>
void print_vec_of_vecs(std::vector<std::vector<T>> const &vec) {
  std::cout << "[" << std::endl;
  for (const auto el : vec) {
    print_vec(el);
  }
  std::cout << "]" << std::endl;
}

template <typename T> T dot(const std::vector<T> x, const std::vector<T> y) {

  if (x.size() != y.size()) {
    throw std::runtime_error("Vector size mismatch in dot product.");
  }
  const T dp = std::inner_product(x.begin(), x.end(), y.begin(), 0);
  return dp;
};
