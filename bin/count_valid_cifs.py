import sys
sys.path.append(".")
import gzip
import queue
import numpy as np
import pandas as pd
from tqdm import tqdm
import multiprocessing as mp
import signal
from lib import is_atom_site_multiplicity_consistent, is_space_group_consistent, extract_space_group_symbol, \
    extract_data_formula, replace_symmetry_operators, bond_length_reasonableness_score, is_formula_consistent, \
    get_cif_tokenizer, extract_numeric_property, get_unit_cell_volume

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


def is_valid(generated_cif, bond_length_acceptability_cutoff):
    if not is_formula_consistent(generated_cif):
        return False
    if not is_atom_site_multiplicity_consistent(generated_cif):
        return False
    bond_length_score = bond_length_reasonableness_score(generated_cif)
    if bond_length_score < bond_length_acceptability_cutoff:
        return False
    if not is_space_group_consistent(generated_cif):
        return False
    return True


def eval_cif(progress_queue, task_queue, result_queue):
    is_valid_and_len = []
    tokenizer = get_cif_tokenizer(symmetrized=True, includes_props=True)

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
            # try to calculate the implied volume, to weed out very bad generations;
            #  an exception will be thrown if a value is missing, or the volume is nonsensical
            a = extract_numeric_property(cif, "_cell_length_a")
            b = extract_numeric_property(cif, "_cell_length_b")
            c = extract_numeric_property(cif, "_cell_length_c")
            alpha = extract_numeric_property(cif, "_cell_angle_alpha")
            beta = extract_numeric_property(cif, "_cell_angle_beta")
            gamma = extract_numeric_property(cif, "_cell_angle_gamma")
            get_unit_cell_volume(a, b, c, alpha, beta, gamma)

            gen_len = len(tokenizer.tokenize_cif(cif))

            data_formula = extract_data_formula(cif)

            # add back the symm operators first
            space_group_symbol = extract_space_group_symbol(cif)
            if space_group_symbol is not None and space_group_symbol != "P 1":
                cif = replace_symmetry_operators(cif, space_group_symbol)

            valid = is_valid(cif, bond_length_acceptability_cutoff=1.0)

            is_valid_and_len.append((data_formula, space_group_symbol, valid, gen_len))

            signal.alarm(0)  # cancel the alarm

        except Exception:
            signal.alarm(0)  # cancel the alarm
            pass

        progress_queue.put(1)

    result_queue.put(is_valid_and_len)


if __name__ == '__main__':
    fname_pred = "../out/cif_model_24.evalcifs.pkl.gz"
    out_fname = "../out/cif_model_24.evalcifs.results.csv"
    workers = 2

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

    is_valid_and_lens = []

    while not result_queue.empty():
        is_valid_and_len = result_queue.get()
        is_valid_and_lens.extend(is_valid_and_len)

    n_valid = 0
    valid_gen_lens = []
    results_data = {
        "comp": [],
        "sg": [],
        "is_valid": [],
        "gen_len": [],
    }
    for comp, sg, valid, gen_len in is_valid_and_lens:
        if valid:
            n_valid += 1
            valid_gen_lens.append(gen_len)
        results_data["comp"].append(comp)
        results_data["sg"].append(sg)
        results_data["is_valid"].append(valid)
        results_data["gen_len"].append(gen_len)

    print(f"num valid: {n_valid} / {n} ({n_valid/n:.2f})")
    print(f"longest valid generated length: {np.max(valid_gen_lens):,}")
    print(f"avg. valid generated length: {np.mean(valid_gen_lens):.3f} ± {np.std(valid_gen_lens):.3f}")

    pd.DataFrame(results_data).to_csv(out_fname)
