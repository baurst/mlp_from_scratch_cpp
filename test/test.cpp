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

TEST_CASE("SoftmaxCEWithLogitsGradient", "SoftmaxCEWithLogitsGradient") {
  using namespace Catch::literals;
  const auto predictions = Mat2D<float>(
      5, 10,
      {5.48813504, 7.15189366, 6.02763376, 5.44883183, 4.23654799, 6.45894113,
       4.37587211, 8.91773001, 9.63662761, 3.83441519, 5.68044561, 9.25596638,
       0.71036058, 0.871293,   0.20218397, 8.32619846, 7.78156751, 8.70012148,
       9.78618342, 7.99158564, 1.18274426, 6.39921021, 1.43353287, 9.44668917,
       5.21848322, 4.1466194,  2.64555612, 7.74233689, 4.56150332, 5.68433949,
       6.12095723, 6.16933997, 9.43748079, 6.81820299, 3.59507901, 4.37031954,
       6.97631196, 0.60225472, 6.66766715, 6.7063787,  3.15428351, 3.63710771,
       5.7019677,  4.38601513, 9.88373838, 1.02044811, 2.08876756, 1.61309518,
       6.53108325, 2.53291603});

  const auto labels_one_hot =
      Mat2D<float>(5, 10, {1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
                           0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
                           0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                           0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.});

  const auto dL_dz = Mat2D<float>(
      5, 10,
      {-1.98123911e-01, 9.90410674e-03,  3.21777327e-03,  1.80378577e-03,
       5.36656971e-04,  4.95301737e-03,  6.16885371e-04,  5.79039636e-02,
       1.18828756e-01,  3.58965505e-04,  1.33102670e-03,  -1.52464761e-01,
       9.24072967e-06,  1.08542126e-05,  5.55914458e-06,  1.87595592e-02,
       1.08815914e-02,  2.72656174e-02,  8.07766882e-02,  1.34246240e-02,
       4.02115344e-05,  7.41025281e-03,  -1.99948330e-01, 1.56076069e-01,
       2.27535836e-03,  7.79014003e-04,  1.73636797e-04,  2.83887017e-02,
       1.17957908e-03,  3.62550588e-03,  5.29726643e-03,  5.55986407e-03,
       1.46015748e-01,  -1.89361958e-01, 4.23717931e-04,  9.19940054e-04,
       1.24602758e-02,  2.12475945e-05,  9.15134485e-03,  9.51255389e-03,
       2.25731723e-04,  3.65830981e-04,  2.88428242e-03,  7.73618495e-04,
       -1.11322937e-02, 2.67226581e-05,  7.77758738e-05,  4.83351431e-05,
       6.60873126e-03,  1.21265183e-04});

  const auto ce_layer = SoftmaxCrossEntropyWithLogitsLoss();
  const auto ce_batched = ce_layer.loss(predictions, labels_one_hot);
  const auto ce = ce_batched.reduce_mean();
  //   REQUIRE(ce == 5.948966026306152);
  REQUIRE(ce == 3.4716506004333496_a);

  const auto grad_test = ce_layer.loss_grad(predictions, labels_one_hot);
  REQUIRE(grad_test.get_num_rows() == dL_dz.get_num_rows());
  REQUIRE(grad_test.get_num_cols() == dL_dz.get_num_cols());
  REQUIRE_THAT(grad_test.matrix_data,
               Catch::Approx(dL_dz.matrix_data).epsilon(1.e-5));
}

TEST_CASE("LeakyReluGradient", "LeakyReluGradient") {

  RELUActivationLayer lrelu(0.1f);
  const auto activations = Mat2D<float>(
      5, 10, {3.06193989,  2.03888584,  -3.99773113, 4.19482614,  2.142413,
              4.98847007,  -3.50551695, 3.68126057,  -3.37507065, 1.15559564,
              -3.76180017, 3.48008229,  3.07318959,  0.69100739,  -0.92816703,
              -4.30833005, 1.97428773,  -0.46457317, 2.22055599,  3.66382326,
              4.75521505,  3.55803342,  -4.88285916, -1.40021936, 2.29990562,
              -3.28370323, 0.21036606,  -4.45662012, -3.00003475, -4.81478206,
              2.93697703,  -2.76075312, -1.54648319, 4.28081293,  2.04414402,
              -4.6816107,  -3.35305844, 1.21478401,  0.77228589,  -2.62107179,
              4.34213998,  1.13965956,  0.35632803,  0.89909976,  2.3012203,
              -1.88055005, -1.01778938, -2.90156251, -3.13806994, 4.4437239});
  const auto gradients_exp = Mat2D<float>(
      5, 10, {1.,  1.,  0.1, 1.,  1.,  1.,  0.1, 1.,  0.1, 1.,  0.1, 1., 1.,
              1.,  0.1, 0.1, 1.,  0.1, 1.,  1.,  1.,  1.,  0.1, 0.1, 1., 0.1,
              1.,  0.1, 0.1, 0.1, 1.,  0.1, 0.1, 1.,  1.,  0.1, 0.1, 1., 1.,
              0.1, 1.,  1.,  1.,  1.,  1.,  0.1, 0.1, 0.1, 0.1, 1.});
  std::vector<float> ones(5 * 10, 1.0);
  const auto grads_at_output = Mat2D<float>(5, 10, ones);

  const auto grads_actual =
      lrelu.backward(activations, grads_at_output, Mat2D<float>(1, 1, {0.0f}));

  REQUIRE_THAT(grads_actual.matrix_data,
               Catch::Approx(gradients_exp.matrix_data).epsilon(1.e-5));
}

TEST_CASE("BiasInit", "BiasInit") {

  DenseLayer layer(10, 5);
  std::vector<float> zeros(5, 0.0);

  REQUIRE_THAT(layer.biases.matrix_data, Catch::Approx(zeros).epsilon(1.e-5));
}