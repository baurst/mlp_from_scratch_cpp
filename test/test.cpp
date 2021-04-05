#define CATCH_CONFIG_MAIN
#include "layer.h"
#include "utils.h"
#include <catch2/catch.hpp>

int factorial(int foo) {
  int result = 1;
  for (int i = 1; i <= foo; ++i) {
    result *= i;
  }
  return result;
}

TEST_CASE("Mat2D Tests hadamard_product", "hadamard_product") {
  const auto A = Mat2D<float>(2, 5, {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.});
  const auto B = A.hadamard_product(A);
  const auto B_exp =
      Mat2D<float>(2, 5, {0., 1., 4., 9., 16., 25., 36., 49., 64., 81.});
  REQUIRE_THAT(B.matrix_data, Catch::Approx(B_exp.matrix_data).epsilon(1.e-5));

  REQUIRE_THAT(B.hadamard_product(A).matrix_data,
               Catch::Approx(A.hadamard_product(B).matrix_data).epsilon(1.e-5));
}

TEST_CASE("Mat2D Tests dot_product", "dot_product") {
  auto A = Mat2D<float>(2, 5, {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.});
  auto B = A.dot_product(A.transpose());
  auto B_exp = Mat2D<float>(2, 2, {30., 80., 80., 255.});
  REQUIRE(B.get_num_rows() == B_exp.get_num_rows());
  REQUIRE(B.get_num_cols() == B_exp.get_num_cols());
  REQUIRE_THAT(B.matrix_data, Catch::Approx(B_exp.matrix_data).epsilon(1.e-5));

  A = Mat2D<float>(1, 10, {0., 1., 2., 3., 4., 5., 6., 7., 8., 9.});
  B = A.dot_product(A.transpose());
  B_exp = Mat2D<float>(1, 1, {285.});
  REQUIRE(B.get_num_rows() == B_exp.get_num_rows());
  REQUIRE(B.get_num_cols() == B_exp.get_num_cols());
  REQUIRE_THAT(B.matrix_data, Catch::Approx(B_exp.matrix_data).epsilon(1.e-5));

  B = A.transpose().dot_product(A);
  B_exp = Mat2D<float>(
      10, 10, {0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  1.,  2.,
               3.,  4.,  5.,  6.,  7.,  8.,  9.,  0.,  2.,  4.,  6.,  8.,  10.,
               12., 14., 16., 18., 0.,  3.,  6.,  9.,  12., 15., 18., 21., 24.,
               27., 0.,  4.,  8.,  12., 16., 20., 24., 28., 32., 36., 0.,  5.,
               10., 15., 20., 25., 30., 35., 40., 45., 0.,  6.,  12., 18., 24.,
               30., 36., 42., 48., 54., 0.,  7.,  14., 21., 28., 35., 42., 49.,
               56., 63., 0.,  8.,  16., 24., 32., 40., 48., 56., 64., 72., 0.,
               9.,  18., 27., 36., 45., 54., 63., 72., 81.});
  REQUIRE(B.get_num_rows() == B_exp.get_num_rows());
  REQUIRE(B.get_num_cols() == B_exp.get_num_cols());
  REQUIRE_THAT(B.matrix_data, Catch::Approx(B_exp.matrix_data).epsilon(1.e-5));
}

TEST_CASE("Reduce axis", "reduce_(max|sum)_axis") {
  // MAX
  const auto A = Mat2D<float>(
      10, 10, {0.,  8.,  16., 24., 32., 40., 48., 56., 64., 72., 0.,  7.,  14.,
               21., 28., 35., 42., 49., 56., 63., 0.,  2.,  4.,  6.,  8.,  10.,
               12., 14., 16., 18., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
               0.,  0.,  9.,  18., 27., 36., 45., 54., 63., 72., 81., 0.,  3.,
               6.,  9.,  12., 15., 18., 21., 24., 27., 0.,  1.,  2.,  3.,  4.,
               5.,  6.,  7.,  8.,  9.,  0.,  5.,  10., 15., 20., 25., 30., 35.,
               40., 45., 0.,  6.,  12., 18., 24., 30., 36., 42., 48., 54., 0.,
               4.,  8.,  12., 16., 20., 24., 28., 32., 36.});

  const auto A_red_0 = A.reduce_max_axis(0);
  const auto A_exp_red_max_0 =
      Mat2D<float>(1, 10, {0., 9., 18., 27., 36., 45., 54., 63., 72., 81.});
  REQUIRE(A_red_0.get_num_rows() == A_exp_red_max_0.get_num_rows());
  REQUIRE(A_red_0.get_num_cols() == A_exp_red_max_0.get_num_cols());
  REQUIRE_THAT(A_red_0.matrix_data,
               Catch::Approx(A_exp_red_max_0.matrix_data).epsilon(1.e-5));

  const auto A_red_1 = A.reduce_max_axis(1);
  const auto A_exp_red_max_1 =
      Mat2D<float>(10, 1, {72., 63., 18., 0., 81., 27., 9., 45., 54., 36.});

  REQUIRE(A_red_1.get_num_rows() == A_exp_red_max_1.get_num_rows());
  REQUIRE(A_red_1.get_num_cols() == A_exp_red_max_1.get_num_cols());
  REQUIRE_THAT(A_red_1.matrix_data,
               Catch::Approx(A_exp_red_max_1.matrix_data).epsilon(1.e-5));

  // SUM
  const auto A_exp_sum_0 = Mat2D<float>(
      1, 10, {0., 45., 90., 135., 180., 225., 270., 315., 360., 405.});
  const auto A_test_sum_0 = A.reduce_sum_axis(0);
  REQUIRE(A_test_sum_0.get_num_rows() == A_exp_sum_0.get_num_rows());
  REQUIRE(A_test_sum_0.get_num_cols() == A_exp_sum_0.get_num_cols());
  REQUIRE_THAT(A_test_sum_0.matrix_data,
               Catch::Approx(A_exp_sum_0.matrix_data).epsilon(1.e-5));

  const auto A_test_sum_1 = A.reduce_sum_axis(1);

  const auto A_exp_sum_1 = Mat2D<float>(
      10, 1, {360., 315., 90., 0., 405., 135., 45., 225., 270., 180.});

  REQUIRE(A_test_sum_1.get_num_rows() == A_exp_sum_1.get_num_rows());
  REQUIRE(A_test_sum_1.get_num_cols() == A_exp_sum_1.get_num_cols());
  REQUIRE_THAT(A_test_sum_1.matrix_data,
               Catch::Approx(A_exp_sum_1.matrix_data).epsilon(1.e-5));
}

TEST_CASE("Addition/Subtraction", "Addition/Subtraction") {

  const auto A = Mat2D<float>(
      10, 10, {0.,  8.,  16., 24., 32., 40., 48., 56., 64., 72., 0.,  7.,  14.,
               21., 28., 35., 42., 49., 56., 63., 0.,  2.,  4.,  6.,  8.,  10.,
               12., 14., 16., 18., 0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,  0.,
               0.,  0.,  9.,  18., 27., 36., 45., 54., 63., 72., 81., 0.,  3.,
               6.,  9.,  12., 15., 18., 21., 24., 27., 0.,  1.,  2.,  3.,  4.,
               5.,  6.,  7.,  8.,  9.,  0.,  5.,  10., 15., 20., 25., 30., 35.,
               40., 45., 0.,  6.,  12., 18., 24., 30., 36., 42., 48., 54., 0.,
               4.,  8.,  12., 16., 20., 24., 28., 32., 36.});
  const auto A_2_exp = Mat2D<float>(
      10, 10,
      {0.,  16., 32., 48., 64., 80.,  96.,  112., 128., 144., 0.,   14.,  28.,
       42., 56., 70., 84., 98., 112., 126., 0.,   4.,   8.,   12.,  16.,  20.,
       24., 28., 32., 36., 0.,  0.,   0.,   0.,   0.,   0.,   0.,   0.,   0.,
       0.,  0.,  18., 36., 54., 72.,  90.,  108., 126., 144., 162., 0.,   6.,
       12., 18., 24., 30., 36., 42.,  48.,  54.,  0.,   2.,   4.,   6.,   8.,
       10., 12., 14., 16., 18., 0.,   10.,  20.,  30.,  40.,  50.,  60.,  70.,
       80., 90., 0.,  12., 24., 36.,  48.,  60.,  72.,  84.,  96.,  108., 0.,
       8.,  16., 24., 32., 40., 48.,  56.,  64.,  72.});
  const auto A_2 = A.add(A);
  REQUIRE(A_2.get_num_rows() == A_2_exp.get_num_rows());
  REQUIRE(A_2.get_num_cols() == A_2_exp.get_num_cols());

  REQUIRE_THAT(A_2.matrix_data,
               Catch::Approx(A_2_exp.matrix_data).epsilon(1.e-5));
  const auto zero_exp = Mat2D<float>(
      10, 10,
      {0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
       0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.});

  const auto zero_test = A.minus(A);

  REQUIRE(zero_exp.get_num_rows() == zero_test.get_num_rows());
  REQUIRE(zero_exp.get_num_cols() == zero_test.get_num_cols());
  REQUIRE_THAT(zero_exp.matrix_data,
               Catch::Approx(zero_test.matrix_data).epsilon(1.e-5));
}

TEST_CASE("Broadcasting Addition/Subtraction", "Broadcasting") {
  const auto row_vec = Mat2D<float>(1, 6, {150., 130., 50., 110., 90., 70.});
  const auto mat = Mat2D<float>(
      6, 6, {0.,  1.,  2.,  3.,  4.,  5.,  12., 13., 14., 15., 16., 17.,
             18., 19., 20., 21., 22., 23., 30., 31., 32., 33., 34., 35.,
             6.,  7.,  8.,  9.,  10., 11., 24., 25., 26., 27., 28., 29.});
  const auto vec_plus_mat_exp =
      Mat2D<float>(6, 6, {150., 131., 52., 113., 94.,  75.,  162., 143., 64.,
                          125., 106., 87., 168., 149., 70.,  131., 112., 93.,
                          180., 161., 82., 143., 124., 105., 156., 137., 58.,
                          119., 100., 81., 174., 155., 76.,  137., 118., 99.});
  const auto vec_plus_mat_test = row_vec.add(mat);

  REQUIRE(vec_plus_mat_test.get_num_rows() == vec_plus_mat_exp.get_num_rows());
  REQUIRE(vec_plus_mat_test.get_num_cols() == vec_plus_mat_exp.get_num_cols());
  REQUIRE_THAT(vec_plus_mat_test.matrix_data,
               Catch::Approx(vec_plus_mat_exp.matrix_data).epsilon(1.e-5));

  const auto mat_plus_vec_test = mat.add(row_vec);
  REQUIRE(mat_plus_vec_test.get_num_rows() == vec_plus_mat_exp.get_num_rows());
  REQUIRE(mat_plus_vec_test.get_num_cols() == vec_plus_mat_exp.get_num_cols());
  REQUIRE_THAT(mat_plus_vec_test.matrix_data,
               Catch::Approx(vec_plus_mat_exp.matrix_data).epsilon(1.e-5));

  const auto mat_plus_vec_T_exp =
      Mat2D<float>(6, 6, {150., 151., 152., 153., 154., 155., 142., 143., 144.,
                          145., 146., 147., 68.,  69.,  70.,  71.,  72.,  73.,
                          140., 141., 142., 143., 144., 145., 96.,  97.,  98.,
                          99.,  100., 101., 94.,  95.,  96.,  97.,  98.,  99.});

  const auto mat_plus_vec_T_test = mat.add(row_vec.transpose());
  REQUIRE(mat_plus_vec_T_test.get_num_rows() ==
          mat_plus_vec_T_exp.get_num_rows());
  REQUIRE(mat_plus_vec_T_test.get_num_cols() ==
          mat_plus_vec_T_exp.get_num_cols());
  REQUIRE_THAT(mat_plus_vec_T_test.matrix_data,
               Catch::Approx(mat_plus_vec_T_exp.matrix_data).epsilon(1.e-5));
}

TEST_CASE("Elementwise Division", "Elementwise Division") {
  const auto some_mat =
      Mat2D<float>(4, 5,
                   {
                       10., 11., 12., 13., 14., 15., 16., 17., 18., 19.,
                       0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.,  9.,
                   });
  const auto divisor = Mat2D<float>(4, 1,
                                    {
                                        0.1,
                                        0.2,
                                        0.3,
                                        0.4,
                                    });
  const auto some_mat_div_10_exp = Mat2D<float>(
      4, 5, {100.,        110., 120., 130., 140.,       75.,        80.,
             85.,         90.,  95.,  0.,   3.33333333, 6.66666667, 10.,
             13.33333333, 12.5, 15.,  17.5, 20.,        22.5});

  const auto some_mat_div_10_test = some_mat.divide_by(divisor);
  REQUIRE(some_mat_div_10_test.get_num_rows() ==
          some_mat_div_10_exp.get_num_rows());
  REQUIRE(some_mat_div_10_test.get_num_cols() ==
          some_mat_div_10_exp.get_num_cols());
  REQUIRE_THAT(some_mat_div_10_test.matrix_data,
               Catch::Approx(some_mat_div_10_exp.matrix_data).epsilon(1.e-5));
}

TEST_CASE("Softmax", "Softmax") {
  const auto mat = Mat2D<float>(
      3, 4, {1.f, 0.f, 2.f, -1.f, 2.f, 4.f, 6.f, 8.f, 3.f, 2.f, 1.f, 0.f});

  const auto softmax_exp = Mat2D<float>(3, 4,
                                        {
                                            0.23688282,
                                            0.08714432,
                                            0.64391426,
                                            0.0320586,
                                            0.00214401,
                                            0.0158422,
                                            0.11705891,
                                            0.86495488,
                                            0.64391426,
                                            0.23688282,
                                            0.08714432,
                                            0.0320586,
                                        });

  const auto softmax_test = softmax(mat);

  REQUIRE(softmax_test.get_num_rows() == softmax_exp.get_num_rows());
  REQUIRE(softmax_test.get_num_cols() == softmax_exp.get_num_cols());
  REQUIRE_THAT(softmax_test.matrix_data,
               Catch::Approx(softmax_exp.matrix_data).epsilon(1.e-5));
}

TEST_CASE("SoftmaxCEWithLogits", "SoftmaxCEWithLogits") {
  const auto predictions = Mat2D<float>(
      5, 10,
      {9.12342578, 0.63358697, 8.06560308, 3.9013097,  2.62197487, 6.23334865,
       6.41026575, 2.81146893, 7.77734698, 4.55262309, 9.60435717, 7.04164476,
       0.23223837, 0.46014417, 2.03215742, 5.46875653, 9.3033918,  7.62148293,
       8.93237563, 8.01205415, 3.71256154, 7.27356444, 3.56967595, 3.91320274,
       5.17653227, 2.23887074, 7.590534,   2.15816133, 6.53402656, 8.40739104,
       7.24587216, 6.99176605, 0.6429947,  5.22698447, 8.62558009, 9.86430273,
       4.34014201, 5.79361825, 4.93217826, 7.08567487, 3.46292678, 3.58536139,
       9.04794016, 8.42519859, 3.06082975, 8.43025028, 6.01086521, 6.6900385,
       7.73285345, 3.04084278});

  const auto labels_one_hot = Mat2D<float>(
      5, 10,
      {
          1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
          0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
          0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
      });
  const auto ce_layer = SoftmaxCrossEntropyWithLogitsLoss();
  const auto ce_batched = ce_layer.loss(predictions, labels_one_hot);
  const auto ce = ce_batched.reduce_mean();
  REQUIRE(ce == 4.318751335144043);
}
