import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


if __name__ == '__main__':
    result_fname = "../data/pyrochlores-dft-results-cif_model_24_unseen.csv"
    out_fname = "../out/pyrochlore_a_dft_vs_gen.pdf"
    font_size = 12

    df = pd.read_csv(result_fname)
    x = df['a_dft']
    y = df['a_cif_avg']

    print(df)

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    matplotlib.rc('font', size=font_size)

    ax.errorbar(x, y, yerr=df['a_cif_std'], fmt='o', ecolor='lightgray', capsize=5)

    m, b = np.polyfit(x, y, 1)
    ax.plot(x, m * x + b, color='red')
    ax.grid(alpha=0.15)

    r2 = r2_score(y, x)
    mae = mean_absolute_error(y, x)

    plt.text(10.35, 10.85, '$R^2$ = {:.2f}'.format(r2), fontsize=font_size)
    plt.text(10.35, 10.8, 'MAE = {:.2f}'.format(mae), fontsize=font_size)

    plt.xlabel('DFT $a$ ($\AA$)')
    plt.ylabel('CrystaLLM $a$ ($\AA$)')

    for i, formula in enumerate(df['formula']):

        if formula == "Ce16Hf16O56":
            xytext = (-57, -14)
        elif formula == "Ce16Mn16O56":
            xytext = (-62, 6)
        elif formula == "La16Mn16O56":
            xytext = (7, 1)
        elif formula == "La16V16O56":
            xytext = (-35, -20)
        elif formula == "Lu16Hf16O56":
            xytext = (7, 12)
        elif formula == "Lu16Zr16O56":
            xytext = (7, -10)
        elif formula == "Pr16Mn16O56":
            xytext = (-62, 8)
        elif formula == "Pr16V16O56":
            xytext = (-40, 8)
        elif formula == "Pr16Hf16O56":
            xytext = (7, 1)
        else:
            raise Exception(f"unknown formula {formula}")

        comp = Composition(formula).reduced_composition.to_latex_string()
        plt.annotate(comp, (x[i], y[i]), xytext=xytext,
                     textcoords='offset points', color='gray', fontsize=font_size)

    plt.savefig(out_fname)
    plt.show()
