import pickle
from pymatgen.core import Composition
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    fname = "../scratches/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.all-COMPS-LEN.pkl"
    out_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.all-template_counts.pkl"

    with open(fname, "rb") as f:
        formulas = pickle.load(f)

    templates = {}

    for formula in tqdm(formulas):
        for comp, sg, _ in formulas[formula]:
            nonreduced_comp = Composition(comp)
            reduced_comp, Z = nonreduced_comp.get_reduced_composition_and_factor()
            reduced_counts = tuple(sorted(reduced_comp.to_reduced_dict.values()))

            key = (reduced_counts, Z, sg)
            if key not in templates:
                templates[key] = 0
            templates[key] += 1

    print(f"number of templates: {len(templates)}")

    with open(out_fname, "wb") as f:
        pickle.dump(templates, f, protocol=pickle.HIGHEST_PROTOCOL)
