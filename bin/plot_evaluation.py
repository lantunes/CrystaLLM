import matplotlib.pyplot as plt
from lib import plot_true_vs_predicted
try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    fname = "../out/cif_model_18.eval.pkl"

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

    plot_true_vs_predicted(ax[2, 1], results["cell_volume"]["true"], results["cell_volume"]["predicted"],
                           outlier_multiplier=2, alpha=0.3, text="cell_volume")

    fig.delaxes(ax[2, 0])
    fig.delaxes(ax[2, 2])

    plt.show()
