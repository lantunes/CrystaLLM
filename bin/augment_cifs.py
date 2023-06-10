import sys
sys.path.append(".")
import gzip
from tqdm import tqdm
import multiprocessing as mp
from queue import Empty
from lib import semisymmetrize_cif, replace_data_formula_with_nonreduced_formula, add_atomic_props_block, \
    round_numbers, extract_formula_units
try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    tot = 0
    while True:
        message = queue.get()
        tot += message
        pbar.update(message)
        if tot == n:
            break


def augment_cif(progress_queue, task_queue, result_queue, oxi, decimal_places):
    augmented_cifs = []

    while not task_queue.empty():
        try:
            cif_str = task_queue.get_nowait()
        except Empty:
            break

        try:
            formula_units = extract_formula_units(cif_str)
            # exclude CIFs with formula units (Z) = 0, which are erroneous
            if formula_units == 0:
                raise Exception()

            cif_str = replace_data_formula_with_nonreduced_formula(cif_str)
            cif_str = semisymmetrize_cif(cif_str)
            cif_str = add_atomic_props_block(cif_str, oxi)
            cif_str = round_numbers(cif_str, decimal_places=decimal_places)
            augmented_cifs.append(cif_str)
        except Exception:
            pass

        progress_queue.put(1)

    result_queue.put(augmented_cifs)


if __name__ == '__main__':
    cifs_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg.pkl.gz"
    out_fname = "../out/orig_cifs_mp_2022_04_12+oqmd_v1_5+nomad_2023_04_30__comp-sg_augm.pkl.gz"
    oxi = False  # whether the CIFs to modify contain oxidation state information
    decimal_places = 4  # the number of decimal places to round the floating point numbers to in the CIF
    workers = 4

    with gzip.open(cifs_fname, "rb") as f:
        cifs = pickle.load(f)

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    for _, cif in cifs:
        task_queue.put(cif)

    watcher = mp.Process(target=progress_listener, args=(progress_queue, len(cifs),))

    processes = [mp.Process(target=augment_cif, args=(progress_queue, task_queue, result_queue, oxi, decimal_places))
                 for _ in range(workers)]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    modified_cifs = []

    while not result_queue.empty():
        modified_cifs.extend(result_queue.get())

    print(f"number of CIFs: {len(modified_cifs)}")

    with gzip.open(out_fname, "wb") as f:
        pickle.dump(modified_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
