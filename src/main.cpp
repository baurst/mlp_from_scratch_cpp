#include "layer.h"
#include "mlp.h"
#include "mnist.h"
#include "utils.h"
#include <iostream>

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

  MNISTDataset train_ds(mnist_train_ds_path, batch_size);
  for (size_t epoch = 0; epoch < num_train_epochs; ++epoch) {

    while (train_ds.hasNext()) {
      const auto [training_input, target_label] = train_ds.next();

      // std::cout << "Input: " << std::endl;
      // std::cout << training_input;

      const auto output = mlp.call(training_input);
      // std::cout << "Output: " << std::endl;
      // std::cout << output;
      std::cout << train_ds.get_progress() << std::endl;
      const auto diff = target_label.minus(output);
      const auto loss_vec = (diff).hadamard_product(diff);
      std::cout << loss_vec << std::endl;
    }
  }
  return 0;
}