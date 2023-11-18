import pandas as pd
from lib import plot_true_vs_predicted
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


if __name__ == '__main__':
    result_fname = "../out/cif_model_35val.evalcifs.results.csv"
    out_fname = "../out/cell_volumes_implied_vs_gen.pdf"

    true_col = "implied_vol"
    gen_col = "gen_vol"

    df = pd.read_csv(result_fname)
    df = df[df.is_valid]

    r2 = r2_score(df[true_col], df[gen_col])
    mae = mean_absolute_error(df[true_col], df[gen_col])
    text = f"$R^2$: {r2:.3f}\nMAE: {mae:.3f}"
    text_coords = (0.05, 0.90)

    fig, ax = plt.subplots(1, 1)

    plot_true_vs_predicted(ax, true_y=df[true_col], predicted_y=df[gen_col],
                           metrics=False, alpha=.75, size=50.,
                           xlabel="Implied Cell Volume ($\mathrm{Å}^3$)", ylabel="Generated Cell Volume ($\mathrm{Å}^3$)",
                           text=text, text_coords=text_coords, rasterize=True)

    ax.set_xlim((2, 3000))
    ax.set_ylim((2, 3000))

    plt.savefig(out_fname)
    plt.show()
