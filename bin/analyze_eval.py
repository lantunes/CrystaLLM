import matplotlib.pyplot as plt
import numpy as np
try:
    import cPickle as pickle
except ImportError:
    import pickle


def remove_outliers(actual_values, predicted_values, multiplier=1.5):

    def get_outlier_bounds(values, multiplier=multiplier):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        return lower_bound, upper_bound

    actual_lower_bound, actual_upper_bound = get_outlier_bounds(actual_values)
    predicted_lower_bound, predicted_upper_bound = get_outlier_bounds(predicted_values)

    filtered_indices = [
        i for i, (actual, predicted) in enumerate(zip(actual_values, predicted_values))
        if actual_lower_bound <= actual <= actual_upper_bound
           and predicted_lower_bound <= predicted <= predicted_upper_bound
    ]

    filtered_actual_values = [actual_values[i] for i in filtered_indices]
    filtered_predicted_values = [predicted_values[i] for i in filtered_indices]

    return filtered_actual_values, filtered_predicted_values


def plot_true_vs_predicted(ax, true_y, predicted_y, xlabel="true", outlier_multiplier=None, ylabel="predicted",
                           min_extra=1, max_extra=1, text=None,
                           alpha=None, title=None, trim_lims=False, size=3, color="lightblue",
                           legend_labels=None, legend_fontsize=6, legend_title=None, legend_loc=None):

    if outlier_multiplier is not None:
        true_y, predicted_y = remove_outliers(true_y, predicted_y, outlier_multiplier)

    line_start = np.min([np.min(true_y), np.min(predicted_y)]) - min_extra
    line_end = np.max([np.max(true_y), np.max(predicted_y)]) + max_extra

    scatter = ax.scatter(true_y, predicted_y, s=size, linewidth=0.1, edgecolor="black", c=color, alpha=alpha)
    ax.plot([line_start, line_end], [line_start, line_end], 'k-', linewidth=0.35)
    if trim_lims:
        ax.set_xlim(np.min(true_y) - min_extra, np.max(true_y) + max_extra)
        ax.set_ylim(np.min(predicted_y) - min_extra, np.max(predicted_y) + max_extra)
    else:
        ax.set_xlim(line_start, line_end)
        ax.set_ylim(line_start, line_end)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)

    if title is not None:
        ax.set_title(title)

    if text:
        ax.text(0.01, 0.92, text, transform=ax.transAxes)

    if legend_labels is not None:
        leg = ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, fontsize=legend_fontsize,
                         markerscale=0.5, loc=legend_loc)
        if legend_title is not None:
            leg.set_title(legend_title, prop={'size': legend_fontsize})


if __name__ == '__main__':
    fname = "../out/cif_model_10.eval.pkl"

    with open(fname, "rb") as f:
        results = pickle.load(f)

    fig, ax = plt.subplots(3, 3)


    plot_true_vs_predicted(ax[0, 0], results["cell_length_a"]["true"], results["cell_length_a"]["predicted"],
                           outlier_multiplier=2, alpha=0.3, text="cell_length_a")

    plot_true_vs_predicted(ax[0, 1], results["cell_length_b"]["true"], results["cell_length_b"]["predicted"],
                           outlier_multiplier=2, alpha=0.3, text="cell_length_b")

    plot_true_vs_predicted(ax[0, 2], results["cell_length_c"]["true"], results["cell_length_c"]["predicted"],
                           outlier_multiplier=2, alpha=0.3, text="cell_length_c")

    plot_true_vs_predicted(ax[1, 0], results["cell_angle_alpha"]["true"], results["cell_angle_alpha"]["predicted"],
                           text="cell_angle_alpha")

    plot_true_vs_predicted(ax[1, 1], results["cell_angle_beta"]["true"], results["cell_angle_beta"]["predicted"],
                           text="cell_angle_beta")

    plot_true_vs_predicted(ax[1, 2], results["cell_angle_gamma"]["true"], results["cell_angle_gamma"]["predicted"],
                           text="cell_angle_gamma")

    plt.show()

    print()
