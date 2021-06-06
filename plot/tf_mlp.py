import tensorflow as tf
import numpy as np
import argparse


@tf.function
def train_step(model, optimizer, x_batch_train, y_batch_train):
    with tf.GradientTape() as tape:
        logits = model(x_batch_train, training=True)
        loss_value = tf.nn.sparse_softmax_cross_entropy_with_logits(
            tf.cast(y_batch_train, tf.int32), logits
        )
        loss_value = tf.reduce_mean(loss_value)
    grads = tape.gradient(loss_value, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    return loss_value


def run_validation(model, dataset):
    acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
    for x_batch_val, y_batch_val in dataset:
        val_logits = model(x_batch_val, training=False)
        acc_metric.update_state(y_batch_val, val_logits)
    val_acc = acc_metric.result()
    return val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logfile", type=str, default=None)
    args = parser.parse_args()

    batch_size = 64
    learning_rate = 0.05
    num_online_val_steps = 20
    log_loss_every_n_steps = 100
    epochs = 10

    leaky_relu_activation = tf.keras.layers.LeakyReLU(alpha=0.1)
    kernel_initializer = tf.keras.initializers.RandomUniform(
        minval=-0.1, maxval=0.1, seed=None
    )

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                50,
                activation=leaky_relu_activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.keras.layers.Dense(
                25,
                activation=leaky_relu_activation,
                kernel_initializer=kernel_initializer,
            ),
            tf.keras.layers.Dense(
                10, activation=None, kernel_initializer=kernel_initializer
            ),
        ]
    )

    optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate)

    (imgs_train, labels_train), (
        imgs_test,
        labels_test,
    ) = tf.keras.datasets.mnist.load_data()
    imgs_train = (np.reshape(imgs_train, (-1, 784)) / 255.0 - 0.5).astype(np.float32)
    imgs_test = (np.reshape(imgs_test, (-1, 784)) / 255.0 - 0.5).astype(np.float32)

    num_val_samples = num_online_val_steps * batch_size
    imgs_val = imgs_train[-num_val_samples:]
    labels_val = labels_train[-num_val_samples:]
    imgs_train = imgs_train[:-num_val_samples]
    labels_train = labels_train[:-num_val_samples]

    train_dataset = (
        tf.data.Dataset.from_tensor_slices((imgs_train, labels_train))
        .shuffle(buffer_size=1024)
        .batch(batch_size)
    )

    val_dataset = tf.data.Dataset.from_tensor_slices((imgs_val, labels_val)).batch(
        batch_size
    )

    test_dataset = tf.data.Dataset.from_tensor_slices((imgs_test, labels_test)).batch(
        batch_size
    )

    global_step = 0
    online_val_accs = []
    log_steps = []
    loss_vals = []
    for epoch in range(epochs):
        # slight learning rate decay
        optimizer.learning_rate = learning_rate * 0.775 ** epoch
        print(
            "\nStarting epoch {0} - learning rate: {1}".format(
                epoch, optimizer.learning_rate.numpy()
            )
        )

        for (train_imgs, labels_gt) in train_dataset:
            loss_value = train_step(model, optimizer, train_imgs, labels_gt)

            if global_step % log_loss_every_n_steps == 0:
                loss_vals.append(loss_value)
                print("Step: {0} - Loss: {1:.4f}".format(global_step, loss_value))
                online_val_acc = run_validation(model, val_dataset)

                log_steps.append(global_step)
                online_val_accs.append(online_val_acc)
                print(
                    "Step: {0} - Online VAL Accuracy: {1:.4f}".format(
                        global_step,
                        online_val_acc,
                    )
                )

                print(
                    "Step: {0} - Online VAL ON TRAIN Accuracy: {1:.4f}".format(
                        global_step,
                        run_validation(
                            model,
                            train_dataset.take(num_online_val_steps),
                        ),
                    )
                )
            global_step += 1

    test_acc = run_validation(model, test_dataset)
    print("\n\nStep: {0} - Test Accuracy: {1:.4f}".format(global_step, test_acc))
    if args.logfile is not None:
        np.savez(
            args.logfile,
            loss=np.array(loss_vals),
            steps=np.array(log_steps),
            val_accuracy=np.array(online_val_accs),
            test_accuracy=np.array(test_acc),
        )


if __name__ == "__main__":
    main()
