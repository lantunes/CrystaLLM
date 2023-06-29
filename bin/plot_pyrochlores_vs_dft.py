import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pymatgen.core import Composition
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


if __name__ == '__main__':
    result_fname = "../data/pyrochlores-dft-results-cif_model_20_unseen.csv"
    out_fname = "../out/pyrochlore_a_dft_vs_gen.pdf"

    df = pd.read_csv(result_fname)

    print(df)

    plt.figure(figsize=(12, 6))

    plt.errorbar(df['a_dft'], df['a_cif_avg'], yerr=df['a_cif_std'], fmt='o', ecolor='lightgray', capsize=5)

    m, b = np.polyfit(df['a_dft'], df['a_cif_avg'], 1)
    plt.plot(df['a_dft'], m * df['a_dft'] + b, color='red')

    r2 = r2_score(df['a_cif_avg'], df['a_dft'])
    mae = mean_absolute_error(df['a_cif_avg'], df['a_dft'])

    plt.text(10.35, 10.85, '$R^2$ = {:.2f}'.format(r2), fontsize=12)
    plt.text(10.35, 10.8, 'MAE = {:.2f}'.format(mae), fontsize=12)

    plt.xlabel('DFT (Å)')
    plt.ylabel('CrystaLLM (Å)')
    plt.title('Cell parameter $a$: DFT vs CrystaLLM (mean value over 3 attempts)')

    for i, formula in enumerate(df['formula']):
        xytext = (-54, -16) if formula == "Ce16Hf16O56" else (7, 1)
        comp = Composition(formula).reduced_composition.to_latex_string()
        plt.annotate(comp, (df['a_dft'][i], df['a_cif_avg'][i]), xytext=xytext,
                     textcoords='offset points', color='gray')

    plt.savefig(out_fname)
    plt.show()
