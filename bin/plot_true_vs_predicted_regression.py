import numpy as np
from lib import plot_true_vs_predicted
import matplotlib.pyplot as plt


if __name__ == '__main__':
    fname = "../out/vpfu_model_2.predictions.csv"

    # Load the CSV file
    data = np.loadtxt(fname, delimiter=",")

    # Separate the predictions and true values
    predictions = data[:, 0]
    true_values = data[:, 1]

    fig, ax = plt.subplots(1, 1)

    n_outliers_removed = plot_true_vs_predicted(ax, true_values, predictions, outlier_multiplier=7)
    orig_count = len(predictions)
    ax.set_title(f"outliers removed: {n_outliers_removed}/{orig_count} "
                 f"({(n_outliers_removed / orig_count) * 100:.1f}%)")
    plt.show()
