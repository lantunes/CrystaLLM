from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    fname = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_nosymm.pkl"
    out_file = "../out/oqmd_v1_5_matproj_all_2022_04_12_symmvars.cif_nosymm.pkl"
    symmetrize = False

    with open(fname, "rb") as f:
        cifs_raw = pickle.load(f)

    variant_cifs = []

    for cif in tqdm(cifs_raw):
        structure = Structure.from_str(cif, fmt="cif")

        sga = SpacegroupAnalyzer(structure)

        # get the symmetry operations
        symmetry_ops = sga.get_symmetry_operations()

        # apply the symmetry operations to generate unique structures
        unique_structures = []
        structure_matcher = StructureMatcher()

        for op in symmetry_ops:
            transformed_structure = structure.copy()
            transformed_structure.apply_operation(op)

            # check if the transformed structure is unique
            is_unique = True
            for existing_structure in unique_structures:
                if structure_matcher.fit(transformed_structure, existing_structure):
                    is_unique = False
                    break

            if is_unique:
                unique_structures.append(transformed_structure)

        for unique_structure in unique_structures:
            variant_cif = CifWriter(unique_structure, symprec=0.001 if symmetrize else None)
            variant_cifs.append(variant_cif)

    with open(out_file, 'wb') as f:
        pickle.dump(variant_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
