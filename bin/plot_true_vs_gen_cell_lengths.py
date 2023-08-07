import pandas as pd
from lib import plot_true_vs_predicted
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


if __name__ == '__main__':
    result_fname = "../data/cif_model_24.evalresults-sg.first_match_lengths.csv"
    out_fname = "../out/cell_lengths_true_vs_gen.pdf"

    df = pd.read_csv(result_fname)

    true_col = "true_lengths"
    gen_col = "gen_lengths"

    r2 = r2_score(df[true_col], df[gen_col])
    mae = mean_absolute_error(df[true_col], df[gen_col])
    text = f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}"
    text_coords = (0.05, 0.90)

    fig, ax = plt.subplots(1, 1)

    plot_true_vs_predicted(ax, true_y=df[true_col], predicted_y=df[gen_col],
                           metrics=False, alpha=1.,
                           xlabel="True Cell Length (Å)", ylabel="Generated Cell Length (Å)",
                           text=text, text_coords=text_coords, rasterize=True)

    plt.savefig(out_fname)
    plt.show()
