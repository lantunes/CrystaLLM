import gzip
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib_venn import venn3
from pymatgen.core import Composition

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def extract_formula(cif):
    lines = cif.split("\n")
    for line in lines:
        if line.startswith("_chemical_formula_sum"):
            return line.split("_chemical_formula_sum")[1].replace("'", "")


if __name__ == '__main__':
    cifs_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30.pkl.gz"
    out_fname = "../out/cif_counts.pdf"

    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    mp_reduced_formulas = set()
    oqmd_reduced_formulas = set()
    nomad_reduced_formulas = set()

    for id, cif in tqdm(cifs):
        formula = extract_formula(cif)
        comp = Composition(formula)

        if id.startswith("MP"):
            mp_reduced_formulas.add(comp.reduced_formula)
        elif id.startswith("OQMD"):
            oqmd_reduced_formulas.add(comp.reduced_formula)
        elif id.startswith("NOMAD"):
            nomad_reduced_formulas.add(comp.reduced_formula)

    print(f"MP # unique reduced formulas: {len(mp_reduced_formulas)}")
    print(f"OQMD # unique reduced formulas: {len(oqmd_reduced_formulas)}")
    print(f"NOMAD # unique reduced formulas: {len(nomad_reduced_formulas)}")

    print(f"MP-NOMAD intersection count: {len(mp_reduced_formulas.intersection(nomad_reduced_formulas))}")
    print(f"OQMD-NOMAD intersection count: {len(oqmd_reduced_formulas.intersection(nomad_reduced_formulas))}")
    print(f"MP-OQMD intersection count: {len(mp_reduced_formulas.intersection(oqmd_reduced_formulas))}")

    venn3([mp_reduced_formulas, oqmd_reduced_formulas, nomad_reduced_formulas], ("MP", "OQMD", "NOMAD"))

    plt.savefig(out_fname)
    plt.show()
