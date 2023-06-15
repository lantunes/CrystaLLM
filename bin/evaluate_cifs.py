import sys
sys.path.append(".")
import gzip
import queue
import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import signal
from lib import is_atom_site_multiplicity_consistent, is_space_group_consistent, extract_space_group_symbol, \
    replace_symmetry_operators, bond_length_reasonableness_score

try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def alarm_handler(signum, frame):
    """
    NOTE: This only works on UNIX systems
    """
    raise TimeoutError("Processing CIF timed out")


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    tot = 0
    while True:
        message = queue.get()
        tot += message
        pbar.update(message)
        if tot == n:
            break


def eval_cif(progress_queue, task_queue, result_queue):
    n_atom_site_multiplicity_consistent = 0
    n_space_group_consistent = 0
    bond_length_reasonableness_scores = []

    while not task_queue.empty():
        try:
            cif = task_queue.get_nowait()
        except queue.Empty:
            break

        # set the alarm: if the code doesn't complete in the
        #  given time before the alarm is cancelled, am exception will be raised
        signal.signal(signal.SIGALRM, alarm_handler)
        signal.alarm(90)  # time limit in seconds

        try:
            # add back the symm operators first
            space_group_symbol = extract_space_group_symbol(cif)
            if space_group_symbol is not None and space_group_symbol != "P 1":
                cif = replace_symmetry_operators(cif, space_group_symbol)

            if is_atom_site_multiplicity_consistent(cif):
                n_atom_site_multiplicity_consistent += 1

            if is_space_group_consistent(cif):
                n_space_group_consistent += 1

            score = bond_length_reasonableness_score(cif)
            bond_length_reasonableness_scores.append(score)

            signal.alarm(0)  # cancel the alarm

        except Exception:
            signal.alarm(0)  # cancel the alarm
            pass

        progress_queue.put(1)

    result_queue.put((n_atom_site_multiplicity_consistent, n_space_group_consistent, bond_length_reasonableness_scores))


if __name__ == '__main__':
    fname_pred = "../out/cif_model_20.evalcifs.pkl.gz"
    workers = 4

    with gzip.open(fname_pred, "rb") as f:
        cifs_pred = pickle.load(f)

    n_space_group_consistent = 0
    n_atom_site_multiplicity_consistent = 0
    bond_length_reasonableness_scores = []

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    n = 0
    for cifs in cifs_pred:
        if type(cifs) == str:
            cifs = [cifs]
        for cif in cifs:
            n += 1
            task_queue.put(cif)

    watcher = mp.Process(target=progress_listener, args=(progress_queue, n,))

    processes = [mp.Process(target=eval_cif, args=(progress_queue, task_queue, result_queue)) for _ in range(workers)]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    n_atom_site_multiplicity_consistent = 0
    n_space_group_consistent = 0
    bond_length_reasonableness_scores = []

    while not result_queue.empty():
        n_atom_site_occ, n_space_group, scores = result_queue.get()
        n_atom_site_multiplicity_consistent += n_atom_site_occ
        n_space_group_consistent += n_space_group
        bond_length_reasonableness_scores.extend(scores)

    print(f"space_group_consistent: {n_space_group_consistent}/{n} ({n_space_group_consistent/n:.3f})\n "
          f"atom_site_multiplicity_consistent: "
          f"{n_atom_site_multiplicity_consistent}/{n} ({n_atom_site_multiplicity_consistent/n:.3f})\n "
          f"bond length reasonableness score: "
          f"{np.mean(bond_length_reasonableness_scores):.4f} ± {np.std(bond_length_reasonableness_scores):.4f}")
