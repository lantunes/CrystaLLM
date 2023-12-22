import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


if __name__ == '__main__':
    out_fname = "../out/cell_volumes_vs_gen.pdf"

    result_fname = "../data/cif_model_24.evalresults-sg.first_match_volume.csv"
    true_col = "true_volume"
    gen_col = "gen_volume"
    df = pd.read_csv(result_fname)
    r2 = r2_score(df[true_col], df[gen_col])
    mae = mean_absolute_error(df[true_col], df[gen_col])
    text_true = f"True: $R^2$: {r2:.3f}, MAE: {mae:.3f} $\mathrm{{Å}}^3$"

    result_fname = "../out/cif_model_35val.evalcifs.results.csv"
    true_col = "implied_vol"
    gen_col = "gen_vol"
    df2 = pd.read_csv(result_fname)
    df2 = df2[df2.is_valid]
    r2 = r2_score(df2[true_col], df2[gen_col])
    mae = mean_absolute_error(df2[true_col], df2[gen_col])
    text_implied = f"Implied: $R^2$: {r2:.3f}, MAE: {mae:.3f} $\mathrm{{Å}}^3$"

    xlabel = "True or Implied Cell Volume ($\mathrm{Å}^3$)"
    ylabel = "Generated Cell Volume ($\mathrm{Å}^3$)"

    fig, ax = plt.subplots(1, 1)

    true_y = df["true_volume"]
    predicted_y = df["gen_volume"]
    line_start = np.min([np.min(true_y), np.min(predicted_y)]) - 1
    line_end = np.max([np.max(true_y), np.max(predicted_y)]) + 1

    ax.scatter(true_y, predicted_y, s=20., linewidth=0.1, edgecolor="black", c="skyblue", alpha=1., label=text_true).set_rasterized(True)

    true_y = df2["implied_vol"]
    predicted_y = df2["gen_vol"]
    scatter = ax.scatter(true_y, predicted_y, s=20., linewidth=0.1, edgecolor="black", c="red", alpha=1., label=text_implied).set_rasterized(True)

    ax.plot([line_start, line_end], [line_start, line_end], 'k-', linewidth=0.35)

    ax.set_xlim(line_start, line_end)
    ax.set_ylim(line_start, line_end)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim((2, 3000))
    ax.set_ylim((2, 3000))

    ax.text(-0.135, 1.0, "b", transform=ax.transAxes, ha="center", va="center", rotation="horizontal",
            fontsize=23, fontweight="bold", fontname="Arial")

    plt.legend()
    plt.savefig(out_fname, dpi=500)
    plt.show()
