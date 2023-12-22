import pandas as pd
from lib import plot_true_vs_predicted
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error


if __name__ == '__main__':
    result_fname = "../data/cif_model_24.evalresults-sg.first_match_volume.csv"
    out_fname = "../out/cell_volumes_true_vs_gen.pdf"

    true_col = "true_volume"
    gen_col = "gen_volume"

    df = pd.read_csv(result_fname)

    r2 = r2_score(df[true_col], df[gen_col])
    mae = mean_absolute_error(df[true_col], df[gen_col])
    text = f"$R^2$: {r2:.3f}\nMAE: {mae:.3f} $\mathrm{{Å}}^3$"
    text_coords = (0.05, 0.90)

    fig, ax = plt.subplots(1, 1)

    plot_true_vs_predicted(ax, true_y=df[true_col], predicted_y=df[gen_col],
                           metrics=False, alpha=1.,
                           xlabel="True Cell Volume ($\mathrm{Å}^3$)", ylabel="Generated Cell Volume ($\mathrm{Å}^3$)",
                           text=text, text_coords=text_coords, rasterize=True, show_grid=False)

    ax.set_xlim((2, 3000))
    ax.set_ylim((2, 3000))

    plt.savefig(out_fname)
    plt.show()
