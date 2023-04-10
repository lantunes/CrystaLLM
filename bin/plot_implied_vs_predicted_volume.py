import matplotlib.pyplot as plt
from lib import plot_true_vs_predicted, get_unit_cell_volume
try:
    import cPickle as pickle
except ImportError:
    import pickle


def get_implied_volumes(predicted_a, predicted_b, predicted_c, predicted_alpha, predicted_beta, predicted_gamma):
    implied_volumes = []
    assert len(predicted_a) == len(predicted_b) == len(predicted_c) == \
           len(predicted_alpha) == len(predicted_beta) == len(predicted_gamma)

    for i in range(len(predicted_a)):
        a = predicted_a[i]
        b = predicted_b[i]
        c = predicted_c[i]
        alpha = predicted_alpha[i]
        beta = predicted_beta[i]
        gamma = predicted_gamma[i]
        implied_volumes.append(get_unit_cell_volume(a, b, c, alpha, beta, gamma))

    return implied_volumes


if __name__ == '__main__':
    fname = "../out/cif_model_15.eval.pkl"

    with open(fname, "rb") as f:
        results = pickle.load(f)

    fig, ax = plt.subplots(1, 3)


    # implied cell volume
    implied_volumes = get_implied_volumes(
        predicted_a=results["cell_length_a"]["predicted"],
        predicted_b=results["cell_length_b"]["predicted"],
        predicted_c=results["cell_length_c"]["predicted"],
        predicted_alpha=results["cell_angle_alpha"]["predicted"],
        predicted_beta=results["cell_angle_beta"]["predicted"],
        predicted_gamma=results["cell_angle_gamma"]["predicted"]
    )

    orig_count = len(results["cell_length_a"]["predicted"])

    n_outliers_removed = plot_true_vs_predicted(ax[0], implied_volumes, results["cell_volume"]["predicted"],
                                                text="cell_volume implied")
    ax[0].set_title(f"outliers removed: {n_outliers_removed}/{orig_count} "
                    f"({(n_outliers_removed / orig_count) * 100:.1f}%)")

    n_outliers_removed = plot_true_vs_predicted(ax[1], implied_volumes, results["cell_volume"]["predicted"],
                                                outlier_multiplier=10, text="cell_volume implied")
    ax[1].set_title(f"outliers removed: {n_outliers_removed}/{orig_count} "
                    f"({(n_outliers_removed / orig_count) * 100:.1f}%)")

    n_outliers_removed = plot_true_vs_predicted(ax[2], implied_volumes, results["cell_volume"]["predicted"],
                                                outlier_multiplier=5, text="cell_volume implied")
    ax[2].set_title(f"outliers removed: {n_outliers_removed}/{orig_count} "
                    f"({(n_outliers_removed / orig_count) * 100:.1f}%)")

    plt.show()
