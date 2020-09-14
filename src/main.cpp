#include "layer.h"
#include "mlp.h"
#include "mnist.h"
#include "utils.h"
#include <iomanip>
#include <iostream>

float progress(const size_t counter, const size_t ds_size) {
  return static_cast<float>(counter) / static_cast<float>(ds_size);
}
int main(int argc, char *argv[]) {
  std::string mnist_train_ds_path =
      "/lhome/baurst/priv_projects/cpp_mlp/data/mnist_train.csv";
  if (argc == 1) {
    std::cout << "No path passed to dataset, using default path"
              << mnist_train_ds_path << std::endl;
    std::cout
        << "Pass path to mnist csv root dir as argument if you wish to use "
           "another location."
        << std::endl;

  } else if (argc == 2) {
    mnist_train_ds_path = static_cast<std::string>(argv[1]);
    std::cout << "Using mnist csv root dir " << mnist_train_ds_path
              << std::endl;
  }
  std::vector<size_t> layer_sizes = {5};

  auto mlp = MLP(layer_sizes, 784, 10);

  const size_t num_train_epochs = 1;
  const size_t batch_size = 4;

  const auto train_ds = read_mnist_csv(mnist_train_ds_path, batch_size);
  size_t global_step = 0;
  for (size_t epoch = 0; epoch < num_train_epochs; ++epoch) {
    size_t ds_sample_counter = 0;
    for (const auto &mnist_el : train_ds) {
      const auto [training_input, target_label] = mnist_el;

      // std::cout << "Input: " << std::endl;
      // std::cout << training_input;

      const auto output = mlp.call(training_input);
      // std::cout << "Output: " << std::endl;
      // std::cout << output;
      const auto diff = target_label.minus(output);
      const auto loss_vec = diff.hadamard_product(diff);

      if (global_step % 1000 == 0) {
        std::cout << "Epoch: " << std::setw(3) << std::setprecision(3) << epoch
                  << " - Progress: " << std::setw(5) << std::setprecision(3)
                  << progress(ds_sample_counter, train_ds.size())
                  << " - Loss: " << std::setw(5) << std::setprecision(4)
                  << loss_vec.reduce_sum() << std::endl;
      }
      global_step++;
      ds_sample_counter++;
    }
  }
  return 0;
}