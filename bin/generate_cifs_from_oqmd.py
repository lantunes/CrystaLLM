# NOTE: the qmpy dependency is not included in this project, and this script is here for
#  informational reasons only. Running this script requires an environment with the qmpy
#  dependency installed.
# based on https://github.com/Tony-Y/cgnn/blob/master/tools/oqmd_data.py
from qmpy import *
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifWriter
from tqdm import tqdm
import numpy as np
import queue
import multiprocessing as mp
import gzip

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


def get_valid_entries(max_fe=5):
    entries = Calculation.objects \
        .filter(label__in=['static', 'standard']) \
        .filter(converged=True) \
        .exclude(formationenergy=None) \
        .exclude(entry__duplicate_of=None) \
        .filter(formationenergy__delta_e__lt=max_fe) \
        .values_list('entry__duplicate_of', flat=True) \
        .distinct()
    return list(entries)


def get_calculations(entry_id, type):
    return Calculation.objects \
        .filter(entry__duplicate_of=entry_id) \
        .filter(converged=True) \
        .exclude(formationenergy=None) \
        .filter(label=type).all()


def get_valid_calculation(entry_id):
    c = get_calculations(entry_id, 'static')
    if len(c) == 0:
        c = get_calculations(entry_id, 'standard')
    imin = 0
    if len(c) > 1:
        imin = np.argmin([x.formationenergy_set.first().delta_e for x in c])
    return c[int(imin)]


def pymatgen_structure(cell, atomic_numbers, cartesian_coords, magmoms, magnetic=False):
    if magnetic:
        sp = {"magmom": magmoms}
    else:
        sp = None
    return Structure(lattice=cell,
                     species=atomic_numbers,
                     coords=cartesian_coords,
                     coords_are_cartesian=True,
                     site_properties=sp)


def generate_cifs(progress_queue, task_queue, result_queue):
    cifs = []

    while not task_queue.empty():
        try:
            entry_id, cell, atomic_numbers, cartesian_coords, magmoms, c_magmom = task_queue.get_nowait()
        except queue.Empty:
            break

        try:
            structure = pymatgen_structure(cell, atomic_numbers, cartesian_coords, magmoms, magnetic=c_magmom is not None)
            cif = str(CifWriter(structure, symprec=symprec if symmetrize else None))
            comment = f"# OQMD Entry {entry_id} {structure.composition.reduced_formula} " \
                      f"{structure.composition.formula.replace(' ', '')}\n"
            cif = comment + cif
            cifs.append(cif)

        except Exception:
            pass

        progress_queue.put(1)

    result_queue.put(cifs)


if __name__ == '__main__':
    out_file = "orig_cifs_oqmd_v1_5.pkl.gz"
    symmetrize = True
    symprec = 0.1
    workers = 15

    entries = get_valid_entries()
    print("Total Materials:", len(entries))

    print("getting calculation outputs for entries...")
    calc_outputs = []
    for entry_id in tqdm(entries):
        c = get_valid_calculation(entry_id)
        s = c.output
        calc_outputs.append((entry_id, s.cell, s.atomic_numbers, s.cartesian_coords, s.magmoms, c.magmom))

    print("generating CIFs from calculation outputs...")

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    for calc_output in calc_outputs:
        task_queue.put(calc_output)

    watcher = mp.Process(target=progress_listener, args=(progress_queue, len(calc_outputs),))

    processes = [mp.Process(target=generate_cifs, args=(progress_queue, task_queue, result_queue)) for _ in range(workers)]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    all_cifs = []

    while not result_queue.empty():
        generated_cifs = result_queue.get()
        all_cifs.extend(generated_cifs)

    print(f"CIFs generated: {len(all_cifs):,}")

    with gzip.open(out_file, 'wb') as f:
        pickle.dump(all_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
