#pragma once
#include <algorithm>
#include <iostream>
#include <numeric>
#include <random>
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

enum Initializer { ZEROS, RANDOM_UNIFORM };

template <class T> class Mat2D {
public:
  Mat2D(size_t rows, size_t cols, Initializer = ZEROS);
  Mat2D(std::vector<std::vector<T>> data);
  T &operator()(size_t row_idx, size_t col_idx);
  T operator()(size_t row_idx, size_t col_idx) const;
  Mat2D<T> dot_product(const Mat2D<T> &other) const;
  Mat2D<T> add(const Mat2D<T> &other) const;
  Mat2D<T> transpose() const;
  size_t get_num_rows() const;
  size_t get_num_cols() const;

  template <typename U>
  friend std::ostream &operator<<(std::ostream &os, const Mat2D<U> &);

  std::vector<T> matrix_data;

private:
  size_t num_rows;
  size_t num_cols;
};

template <class T>
Mat2D<T>::Mat2D(size_t num_rows, size_t num_cols, const Initializer init)
    : num_rows(num_rows), num_cols(num_cols), matrix_data(num_rows * num_cols) {
  switch (init) {
  case ZEROS:
    std::fill(this->matrix_data.begin(), this->matrix_data.end(),
              static_cast<T>(0.0));
    break;
  case RANDOM_UNIFORM:
    std::default_random_engine generator;
    std::uniform_real_distribution<T> distribution(-0.5, 0.5);
    std::generate(this->matrix_data.begin(), this->matrix_data.end(),
                  [&]() { return distribution(generator); });
    break;
  }
}

template <class T> size_t Mat2D<T>::get_num_rows() const { return num_rows; }
template <class T> size_t Mat2D<T>::get_num_cols() const { return num_cols; }

template <class T>
Mat2D<T>::Mat2D(std::vector<std::vector<T>> data)
    : num_rows(data.size()), num_cols(data.at(0).size()) {
  matrix_data.resize(num_rows * num_cols);
  for (size_t row_idx = 0; row_idx < num_rows; ++row_idx) {
    for (size_t col_idx = 0; col_idx < num_cols; ++col_idx) {
      this->operator()(row_idx, col_idx) = data[row_idx][col_idx];
    }
  }
}

template <class T> T &Mat2D<T>::operator()(size_t row_idx, size_t col_idx) {
  return matrix_data[row_idx * num_cols + col_idx];
}

template <class T>
T Mat2D<T>::operator()(size_t row_idx, size_t col_idx) const {
  return matrix_data[row_idx * num_cols + col_idx];
}

template <class T> Mat2D<T> Mat2D<T>::dot_product(const Mat2D<T> &other) const {

  Mat2D<T> result(num_cols, other.num_rows);
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t k = 0; k < other.num_cols; ++k) {
      for (size_t j = 0; j < other.num_rows; ++j) {
        result(i, k) = result(i, k) + this->operator()(i, j) * other(j, k);
      }
    }
  }
  return result;
}

template <class T> Mat2D<T> Mat2D<T>::add(const Mat2D<T> &other) const {
  Mat2D<T> result(std::max(other.get_num_rows(), get_num_rows()),
                  std::max(other.get_num_cols(), get_num_cols()));

  bool rows_compatible = other.get_num_rows() == get_num_rows() ||
                         get_num_rows() == 1 || get_num_rows() == 1;

  bool cols_compatible = other.get_num_cols() == get_num_cols() ||
                         get_num_cols() == 1 || get_num_cols() == 1;

  if (!rows_compatible) {
    throw std::runtime_error("Matrix Row dim incompatible.");
  }
  if (!cols_compatible) {
    throw std::runtime_error("Matrix Cols dim incompatible.");
  }

  if (other.get_num_rows() == get_num_rows() &&
      other.get_num_cols() == get_num_cols()) {

    for (size_t row_idx = 0; row_idx < get_num_rows(); ++row_idx) {
      for (size_t col_idx = 0; col_idx < get_num_cols(); ++col_idx) {
        result(row_idx, col_idx) =
            this->operator()(row_idx, col_idx) + other(row_idx, col_idx);
      }
    }
  } else if (other.get_num_rows() == 1 || get_num_rows() == 1 ||
             other.get_num_cols() == 1 || get_num_cols() == 1) {
    const size_t one = 1;
    const size_t zero = 0;
    size_t max_rows = 0;
    size_t max_rows_other = 0;
    if (other.get_num_rows() == get_num_rows()) {
      max_rows = get_num_rows();
      max_rows_other = max_rows;
    } else {
      max_rows = std::max(get_num_rows(), one);
      max_rows_other = std::max(other.get_num_rows(), one);
    }
    size_t max_cols = 0;
    size_t max_cols_other = 0;
    if (other.get_num_cols() == get_num_cols()) {
      max_cols = get_num_cols();
      max_cols_other = max_cols;
    } else {
      max_cols = std::max(get_num_cols(), one);
      max_cols_other = std::max(other.get_num_cols(), one);
    }

    for (size_t row_idx = 0; row_idx < result.get_num_rows(); ++row_idx) {
      for (size_t col_idx = 0; col_idx < result.get_num_cols(); ++col_idx) {

        const T val_a =
            this->operator()(std::clamp(row_idx, zero, max_rows - 1),
                             std::clamp(col_idx, zero, max_cols - 1));
        const T val_b = other(std::clamp(row_idx, zero, max_rows_other - 1),
                              std::clamp(col_idx, zero, max_cols_other - 1));

        result(row_idx, col_idx) = val_a + val_b;
      }
    }
  }
  return result;
}

template <class T> Mat2D<T> Mat2D<T>::transpose() const {
  Mat2D<T> result(num_cols, num_rows);
  for (size_t i = 0; i < num_rows; ++i) {
    for (size_t k = 0; k < num_cols; ++k) {
      result(k, i) = this->operator()(i, k);
    }
  }
  return result;
}

template <typename T>
std::ostream &operator<<(std::ostream &os, const Mat2D<T> &mat) {
  os << "[" << std::endl;
  for (size_t row_idx = 0; row_idx < mat.get_num_rows(); ++row_idx) {
    os << "[";

    for (size_t col_idx = 0; col_idx < mat.get_num_cols(); ++col_idx) {
      os << mat(row_idx, col_idx) << ", ";
    }
    os << "]" << std::endl;
  }
  os << "]" << std::endl;

  return os;
}