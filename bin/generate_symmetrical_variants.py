from pymatgen.core.structure import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.cif import CifWriter
import multiprocessing as mp
from tqdm import tqdm
try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def array_split(arr, num_splits):
    split_size, remainder = divmod(len(arr), num_splits)
    splits = []
    start = 0
    for i in range(num_splits):
        end = start + split_size + (i < remainder)
        splits.append(arr[start:end])
        start = end
    return splits


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)


def get_variants(chunk_of_cifs, queue, symmetrize):
    variants = []
    for cif in chunk_of_cifs:
        try:
            structure = Structure.from_str(cif, fmt="cif")

            queue.put(1)

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
                variants.append(str(variant_cif))
        except Exception as e:
            pass
    return variants


if __name__ == '__main__':
    fname = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_nosymm.pkl"
    out_file = "../out/oqmd_v1_5_matproj_all_2022_04_12_symmvars.cif_nosymm.pkl"
    symmetrize = False
    workers = 2

    with open(fname, "rb") as f:
        cifs_raw = pickle.load(f)

    chunks = array_split(cifs_raw, workers)

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(workers)

    watcher = pool.apply_async(progress_listener, (queue, len(cifs_raw),))

    jobs = []
    for i in range(workers):
        chunk = chunks[i]
        job = pool.apply_async(get_variants, (chunk, queue, symmetrize))
        jobs.append(job)

    variant_cifs = []
    for job in jobs:
        variant_cifs.extend(job.get())

    queue.put("kill")
    pool.close()
    pool.join()

    with open(out_file, 'wb') as f:
        pickle.dump(variant_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
