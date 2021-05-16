#include "layer.h"
#include "mlp.h"
#include "mnist.h"
#include "utils.h"
#include <iomanip>
#include <iostream>

#include <algorithm>
#include <random>

float progress(const size_t counter, const size_t ds_size)
{
  return static_cast<float>(counter) / static_cast<float>(ds_size);
}

float run_validation(
    const MLP &network,
    const std::vector<std::pair<Mat2D<float>, Mat2D<float>>> &dataset,
    const size_t num_val_steps)
{
  size_t val_it_counter = 0;
  size_t num_correct_predictions = 0;
  size_t num_classified_samples = 0;
  for (const auto &mnist_el : dataset)
  {
    if (val_it_counter > num_val_steps)
    {
      break;
    }
    const auto [test_input, target_label] = mnist_el;
    const auto pred = network.predict(test_input);
    const auto label = target_label.argmax(1);
    val_it_counter++;

    auto pred_it = pred.matrix_data.begin();
    auto label_it = label.matrix_data.begin();

    while (pred_it != pred.matrix_data.end() ||
           label_it != label.matrix_data.end())
    {
      if (*pred_it == *label_it)
      {
        num_correct_predictions++;
      }
      ++pred_it;
      ++label_it;
    }

    num_classified_samples += pred.get_num_rows();
  }
  return static_cast<float>(num_correct_predictions) /
         static_cast<float>(num_classified_samples);
}

void log_metric(const float metric, std::string metric_description,
                const size_t global_step)
{
  std::cout << "Step: " << std::setw(3) << std::setprecision(3) << global_step
            << " - " << metric_description << ": " << metric << std::endl;
}

int main(int argc, char *argv[])
{
  std::string mnist_train_ds_path = "";
  std::string mnist_test_ds_path = "";
  if (argc != 3)
  {
    std::cout << std::endl
              << "No paths to dataset given!" << std::endl;
    std::cout << "Usage:" << std::endl
              << "./main path/to/tain.csv path/to/test.csv" << std::endl
              << std::endl;
    return 1;
  }
  else
  {
    mnist_train_ds_path = static_cast<std::string>(argv[1]);
    std::cout << "Using mnist csv train dataset " << mnist_train_ds_path
              << std::endl;
    mnist_test_ds_path = static_cast<std::string>(argv[2]);
    std::cout << "Using mnist csv test dataset " << mnist_test_ds_path
              << std::endl;
  }
  std::vector<size_t> layer_sizes = {50, 25};

  const size_t batch_size = 64;
  const float learning_rate = 0.05;
  const size_t num_train_epochs = 10;

  const size_t num_online_val_steps = 20;
  const size_t log_loss_every_n_steps = 100;

  auto mlp = MLP(layer_sizes, /*num_inputs=*/784, /*num_classes=*/10);
  const auto loss_obj = SoftmaxCrossEntropyWithLogitsLoss();
  const auto full_train_data =
      read_mnist_csv(mnist_train_ds_path, batch_size, -1);
  std::vector<std::pair<Mat2D<float>, Mat2D<float>>> train_ds(
      full_train_data.begin(),
      full_train_data.begin() + full_train_data.size() - num_online_val_steps);
  std::vector<std::pair<Mat2D<float>, Mat2D<float>>> val_ds(
      full_train_data.begin() + full_train_data.size() - num_online_val_steps,
      full_train_data.end());

  const auto test_ds = read_mnist_csv(mnist_test_ds_path, 20, -1);
  const auto online_val_ds =
      read_mnist_csv(mnist_test_ds_path, 20, num_online_val_steps);

  size_t global_step = 0;
  for (size_t epoch = 0; epoch < num_train_epochs; ++epoch)
  {

    const auto learning_rate_decayed = learning_rate * static_cast<float>(std::pow(0.775, epoch));

    for (const auto &mnist_el : train_ds)
    {
      const auto [training_input, target_label] = mnist_el;

      const auto loss = mlp.train(training_input, target_label, loss_obj,
                                  learning_rate_decayed);

      if (global_step % log_loss_every_n_steps == 0)
      {
        log_metric(loss, "Loss", global_step);
        log_metric(run_validation(mlp, online_val_ds, online_val_ds.size()),
                   "Online VAL Accuracy", global_step);
        log_metric(run_validation(mlp, train_ds, num_online_val_steps),
                   "Online VAL ON TRAIN Accuracy", global_step);
      }
      global_step++;
    }
    std::cout << "Epoch: " << std::setw(3) << std::setprecision(3) << epoch
              << " finished! - Running eval..." << std::endl;
  }
  log_metric(run_validation(mlp, test_ds, test_ds.size()), "Test Accuracy",
             global_step);
  return 0;
}
