#pragma once
#include <iostream>
#include <vector>

template <typename T> void print_vec(std::vector<T> const &vec) {
  std::cout << "[";
  for (const auto el : vec) {
    std::cout << el << ',';
  }
  std::cout << "]" << std::endl;
}