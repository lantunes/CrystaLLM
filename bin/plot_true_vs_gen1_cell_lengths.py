import pandas as pd
from lib import plot_true_vs_predicted
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


if __name__ == '__main__':
    result_fname = "../data/cell_lengths_true_vs_gen1_cif_model_20.csv"
    out_fname = "../out/cell_lengths_true_vs_gen1_cif_model_20.pdf"

    df = pd.read_csv(result_fname)

    r2 = r2_score(df['true_lengths'], df['gen1_lengths'])
    mae = mean_absolute_error(df['true_lengths'], df['gen1_lengths'])
    text = f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}"
    text_coords = (0.05, 0.90)

    fig, ax = plt.subplots(1, 1)

    plot_true_vs_predicted(ax, true_y=df['true_lengths'], predicted_y=df['gen1_lengths'],
                           metrics=False, alpha=0.2, title="True vs. Generated Cell Lengths",
                           xlabel="True Cell Lengths (Å)", ylabel="Generated Cell Lengths (Å)",
                           text=text, text_coords=text_coords, rasterize=True)

    ax.set_xlim((2, 20))
    ax.set_ylim((2, 20))

    plt.savefig(out_fname)
    plt.show()
