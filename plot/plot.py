import argparse
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as plticker


def main():
    parser = argparse.ArgumentParser(
        description="Generate validation accuracy curves from logfiles."
    )
    parser.add_argument("--ours_acc_file", required=True)
    parser.add_argument("--tf_log_file", required=True)

    args = parser.parse_args()
    accuracy = np.loadtxt(open(args.ours_acc_file, "rb"), delimiter=";")
    test_accuracy = accuracy[-1, 1]
    accuracy = accuracy[:-1, ...]
    tf_log = np.load(args.tf_log_file)

    _, ax = plt.subplots()
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))
    ln_ours_acc = ax.plot(
        accuracy[..., 0],
        100.0 * accuracy[..., 1],
        label="Validation Accuracy (ours)",
        color="blue",
    )

    ln_tf_acc = ax.plot(
        tf_log["steps"],
        tf_log["val_accuracy"] * 100.0,
        label="Validation Accuracy (tensorflow)",
        color="red",
    )
    ln_test_acc_ours = ax.hlines(
        100.0 * test_accuracy,
        xmin=0,
        xmax=accuracy[-1, 0],
        color="blue",
        label="Final Test Accuracy {:.2f}% (ours)".format(100.0 * test_accuracy),
        linestyles="dotted",
    )
    ln_test_acc_tf = ax.hlines(
        100.0 * tf_log["test_accuracy"],
        xmin=0,
        xmax=tf_log["steps"][-1],
        color="red",
        label="Final Test Accuracy {:.2f}% (tensorflow)".format(
            100.0 * tf_log["test_accuracy"]
        ),
        linestyles="dotted",
    )

    ax.set_xlabel("Steps")
    ax.set_ylabel("Accuracy [%]")

    lns = ln_ours_acc + [ln_test_acc_ours] + ln_tf_acc + [ln_test_acc_tf]
    labs = [plot.get_label() for plot in lns]
    ax.legend(lns, labs, loc="center right")

    plt.show()


if __name__ == "__main__":
    main()
