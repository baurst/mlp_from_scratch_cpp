#pragma once
#include <algorithm>
#include <functional>
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
  Mat2D(const size_t rows, const size_t cols, Initializer = ZEROS);
  Mat2D(std::vector<std::vector<T>> data);
  Mat2D(const Mat2D<T> &other); // copy constructor
  ~Mat2D();
  Mat2D(const size_t num_rows, const size_t num_cols, std::vector<T> data);
  T &operator()(size_t row_idx, size_t col_idx);
  T operator()(size_t row_idx, size_t col_idx) const;
  Mat2D<T> dot_product(const Mat2D<T> &other) const;
  Mat2D<T> add(const Mat2D<T> &other) const;
  Mat2D<T> divide_by(const Mat2D<T> &other) const;
  Mat2D<T> minus(const Mat2D<T> &other) const;
  Mat2D<T> hadamard_product(const Mat2D<T> &other) const;
  Mat2D<T>
  elementwise_combination_w_broadcast(const Mat2D<T> &other,
                                      std::function<T(T, T)> modifier) const;
  T reduce_sum() const;
  Mat2D<T> reduce_sum_axis(const size_t axis) const;
  T reduce_mean() const;
  Mat2D<T> reduce_mean_axis(const size_t axis) const;
  Mat2D<T> transpose() const;
  size_t get_num_rows() const;
  size_t get_num_cols() const;

  template <typename U>
  friend std::ostream &operator<<(std::ostream &os, const Mat2D<U> &);
  Mat2D<T> &operator=(const Mat2D<T> &other); // assignment operator
  template <typename U> Mat2D<T> &operator+(const Mat2D<U> &classObj);
  template <typename U> Mat2D<T> &operator-(const Mat2D<U> &classObj);

  std::vector<T> matrix_data;

private:
  size_t num_rows;
  size_t num_cols;
};

template <class T>
template <typename U>
Mat2D<T> &Mat2D<T>::operator+(const Mat2D<U> &other) {
  // ...
  return this->add(other);
}

template <class T>
template <typename U>
Mat2D<T> &Mat2D<T>::operator-(const Mat2D<U> &other) {
  const Mat2D<T> minus_one(0, 0, std::vector({-1.0}));
  return this->add(other);
}

template <class T>
Mat2D<T>::Mat2D(const size_t num_rows, const size_t num_cols,
                std::vector<T> data)
    : num_rows(num_rows), num_cols(num_cols) {
  matrix_data.resize(num_rows * num_cols);
  matrix_data = data;
}

template <class T>
Mat2D<T>::Mat2D(const Mat2D<T> &other)
    : num_rows(other.num_rows), num_cols(other.num_cols) {
  this->matrix_data = other.matrix_data;
}

template <class T> Mat2D<T>::~Mat2D() {}

template <class T> Mat2D<T> &Mat2D<T>::operator=(const Mat2D<T> &other) {
  if (this != &other) {
    this->num_cols = other.num_cols;
    this->num_rows = other.num_rows;
    this->matrix_data = other.matrix_data;
  }
  return *this;
}

template <class T>
Mat2D<T>::Mat2D(const size_t num_rows, const size_t num_cols,
                const Initializer init)
    : matrix_data(num_rows * num_cols), num_rows(num_rows), num_cols(num_cols) {
  switch (init) {
  case Initializer::ZEROS:
    std::fill(this->matrix_data.begin(), this->matrix_data.end(),
              static_cast<T>(0));
    break;
  case Initializer::RANDOM_UNIFORM:
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

  if (num_cols != other.num_rows) {
    throw std::runtime_error(
        "Dot Product: AxB=C -> A.num_cols != B.num_rows (size mismatch).");
  }

  Mat2D<T> result(num_rows, other.num_cols);
  for (size_t i = 0; i < result.num_rows; ++i) {
    for (size_t k = 0; k < result.num_cols; ++k) {
      for (size_t j = 0; j < other.num_rows; ++j) {
        result(i, k) = result(i, k) + this->operator()(i, j) * other(j, k);
      }
    }
  }
  return result;
}

template <class T> Mat2D<T> Mat2D<T>::add(const Mat2D<T> &other) const {
  return this->elementwise_combination_w_broadcast(other, std::plus<T>());
}

template <class T> T Mat2D<T>::reduce_sum() const {
  return std::accumulate(matrix_data.begin(), matrix_data.end(), T());
}

template <class T> Mat2D<T> Mat2D<T>::reduce_sum_axis(const size_t axis) const {
  const auto num_result_rows = (axis == 0 ? 1 : this->get_num_rows());
  const auto num_result_cols = (axis == 1 ? 1 : this->get_num_cols());
  Mat2D<T> result(num_result_rows, num_result_cols);
  if (axis == 0) {
    for (size_t col_idx = 0; col_idx < this->get_num_cols(); ++col_idx) {
      T col_sum = static_cast<T>(0);
      for (size_t row_idx = 0; row_idx < this->get_num_rows(); ++row_idx) {
        col_sum += this->operator()(row_idx, col_idx);
      }
      result(0, col_idx) = col_sum;
    }
  } else if (axis == 1) {
    for (size_t row_idx = 0; row_idx < this->get_num_rows(); ++row_idx) {
      T row_sum = static_cast<T>(0);
      for (size_t col_idx = 0; col_idx < this->get_num_cols(); ++col_idx) {
        row_sum += this->operator()(row_idx, col_idx);
      }
      result(row_idx, 0) = row_sum;
    }
  } else {
    std::runtime_error("Reduce sum: Axis must be 0 or 1.");
  }
  return result;
}

template <class T> T Mat2D<T>::reduce_mean() const {
  return this->reduce_sum() / static_cast<T>(this->get_num_rows()) /
         static_cast<T>(this->get_num_cols());
}

template <class T>
Mat2D<T> Mat2D<T>::reduce_mean_axis(const size_t axis) const {
  const auto axis_sum = this->reduce_sum_axis(axis);
  const auto divisor =
      (axis == 0 ? this->get_num_cols() : this->get_num_rows());

  const auto axis_mean =
      axis_sum.divide_by(Mat2D<T>(1, 1, {static_cast<T>(1.0 / divisor)}));
  return axis_mean;
}

template <class T> Mat2D<T> Mat2D<T>::minus(const Mat2D<T> &other) const {
  return this->elementwise_combination_w_broadcast(other, std::minus<T>());
}

template <class T> Mat2D<T> Mat2D<T>::divide_by(const Mat2D<T> &other) const {
  return this->elementwise_combination_w_broadcast(other, std::divides<T>());
}

template <class T>
Mat2D<T> Mat2D<T>::hadamard_product(const Mat2D<T> &other) const {
  return this->elementwise_combination_w_broadcast(other, std::multiplies<T>());
}

template <class T>
Mat2D<T> Mat2D<T>::elementwise_combination_w_broadcast(
    const Mat2D<T> &other, std::function<T(T, T)> modifier) const {
  Mat2D<T> result(std::max(other.get_num_rows(), get_num_rows()),
                  std::max(other.get_num_cols(), get_num_cols()));

  bool rows_compatible = other.get_num_rows() == get_num_rows() ||
                         other.get_num_rows() == 1 || get_num_rows() == 1;

  bool cols_compatible = other.get_num_cols() == get_num_cols() ||
                         other.get_num_cols() == 1 || get_num_cols() == 1;

  if (!rows_compatible) {
    throw std::runtime_error("Add: Matrix Row dim incompatible.");
  }
  if (!cols_compatible) {
    throw std::runtime_error("Add: Matrix Cols dim incompatible.");
  }

  if (other.get_num_rows() == get_num_rows() &&
      other.get_num_cols() == get_num_cols()) {

    for (size_t row_idx = 0; row_idx < get_num_rows(); ++row_idx) {
      for (size_t col_idx = 0; col_idx < get_num_cols(); ++col_idx) {
        result(row_idx, col_idx) = modifier(this->operator()(row_idx, col_idx),
                                            other(row_idx, col_idx));
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

        result(row_idx, col_idx) = modifier(val_a, val_b);
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

// void test_dot_prod() {
//   std::vector<std::vector<float>> a{{1, 0}, {0, 1}};
//   std::vector<std::vector<float>> b{{4, 1}, {2, 2}};

//   const Mat2D<float> mat_a(a);
//   const Mat2D<float> mat_b(b);

//   std::cout << mat_a.dot_product(mat_b) << std::endl;
// }