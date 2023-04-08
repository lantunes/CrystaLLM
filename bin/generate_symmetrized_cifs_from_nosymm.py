from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import multiprocessing as mp
from tqdm import tqdm
from lib import array_split
try:
    import cPickle as pickle
except ImportError:
    import pickle

import warnings
warnings.filterwarnings("ignore")


def progress_listener(queue, n):
    pbar = tqdm(total=n)
    while True:
        message = queue.get()
        if message == "kill":
            break
        pbar.update(message)


def get_symm_cifs(chunk_of_cifs, queue):
    symm_cifs = []
    for cif in chunk_of_cifs:
        try:
            queue.put(1)
            struct = Structure.from_str(cif, fmt="cif")
            cif_symm = CifWriter(struct, symprec=0.001)
            symm_cifs.append(str(cif_symm))
        except Exception as e:
            pass
    return symm_cifs


if __name__ == '__main__':
    fname = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif_nosymm.pkl"
    out_file = "../out/oqmd_v1_5_matproj_all_2022_04_12.cif.pkl"
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
        job = pool.apply_async(get_symm_cifs, (chunk, queue))
        jobs.append(job)

    symmetrized_cifs = []
    for job in jobs:
        symmetrized_cifs.extend(job.get())

    queue.put("kill")
    pool.close()
    pool.join()

    with open(out_file, 'wb') as f:
        pickle.dump(symmetrized_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)
