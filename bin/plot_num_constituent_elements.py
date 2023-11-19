from pymatgen.core import Composition
import gzip
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib_venn import venn3

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
    out_fname = "../out/num_constit_elems.pdf"

    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    reduced_comps = set()

    for id, cif in tqdm(cifs):
        formula = extract_formula(cif)
        comp = Composition(formula)

        reduced_comp = comp.reduced_composition
        reduced_comps.add(reduced_comp)

    num_elements = {}
    for r in tqdm(reduced_comps):
        n = len(r.elements)
        if n not in num_elements:
            num_elements[n] = 0
        num_elements[n] += 1

    # print(num_elements)
    # {4: 275423, 3: 458293, 5: 20729, 2: 52915, 6: 1736, 7: 204, 1: 89, 8: 16, 9: 5, 10: 1}

    sorted_num_elements = sorted(num_elements.items(), key=lambda i: i[0])
    counts = [i[1] for i in sorted_num_elements]
    n_elems = [i[0] for i in sorted_num_elements]

    plt.bar(n_elems, counts)
    plt.title("Distribution of reduced compositions by \nthe number of constituent elements")
    plt.ylabel("count")
    plt.xlabel("number of constituent elements")
    plt.xticks(list(range(1, 11)))

    plt.savefig(out_fname)
    plt.show()
