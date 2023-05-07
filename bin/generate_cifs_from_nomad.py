import os.path
from os import listdir
from os.path import isfile, join
import json
from queue import Empty
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
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


def generate_cifs(progress_queue, task_queue, result_queue):
    cifs = []
    no_geom_opt = []
    no_material = []

    while not task_queue.empty():
        try:
            json_fname = task_queue.get_nowait()
        except Empty:
            break

        with open(json_fname, "rt") as f:
            obj = json.load(f)

        for entry in obj["data"]:
            try:

                entry_id = entry["entry_id"]
                archive = entry["archive"]

                if "material" in archive["results"]:
                    chemical_formula_descriptive = archive["results"]["material"]["chemical_formula_descriptive"]
                    chemical_formula_reduced = archive["results"]["material"]["chemical_formula_reduced"]
                    space_group_symbol = archive["results"]["material"]["symmetry"]["space_group_symbol"]

                    if "geometry_optimization" in archive["results"]["properties"]:
                        struct_json = archive["results"]["properties"]["geometry_optimization"]["structure_optimized"]

                        lattice_vectors = struct_json["lattice_vectors"]
                        cartesian_site_positions = struct_json["cartesian_site_positions"]
                        species_at_sites = struct_json["species_at_sites"]
                        # the nomad data sometimes contains species like "Pb_d"
                        species_at_sites = [sp.split("_")[0] for sp in species_at_sites]

                        lattice_vectors = np.array(lattice_vectors) * 1e10
                        cartesian_site_positions = np.array(cartesian_site_positions) * 1e10

                        structure = Structure(lattice=lattice_vectors, species=species_at_sites,
                                              coords=cartesian_site_positions, coords_are_cartesian=True)

                        comment = f"# NOMAD Entry {entry_id} " \
                                  f"{chemical_formula_descriptive} {chemical_formula_reduced} {space_group_symbol}\n"
                        cif_writer = CifWriter(structure, symprec=symprec if symmetrize else None)
                        cif = comment + str(cif_writer)

                        cifs.append(cif)

                    else:
                        no_geom_opt.append(f"{json_fname},{entry_id},{chemical_formula_descriptive},"
                                           f"{chemical_formula_reduced},{space_group_symbol}")
                else:
                    no_material.append(f"{json_fname},{entry_id}")

            except Exception:
                pass

        progress_queue.put(1)

    result_queue.put((cifs, no_geom_opt, no_material))


if __name__ == '__main__':
    src_dir = "../out/nomad_entries_data_2023_04_30"
    out_file = "../out/orig_cifs_nomad_2023_04_30.pkl.gz"
    no_geom_opt_out_file = "../out/nomad_entries_data_2023_04_30_no_geom_opt.csv"
    no_material_out_file = "../out/nomad_entries_data_2023_04_30_no_material.csv"
    symmetrize = True
    symprec = 0.1
    workers = 2

    fnames = [f for f in listdir(src_dir) if isfile(join(src_dir, f))]

    manager = mp.Manager()
    progress_queue = manager.Queue()
    task_queue = manager.Queue()
    result_queue = manager.Queue()

    for fname in fnames:
        task_queue.put(os.path.join(src_dir, fname))

    watcher = mp.Process(target=progress_listener, args=(progress_queue, len(fnames),))

    processes = [mp.Process(target=generate_cifs, args=(progress_queue, task_queue, result_queue)) for _ in range(workers)]
    processes.append(watcher)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    all_cifs = []
    all_no_geom_opts = []
    all_no_materials = []

    while not result_queue.empty():
        generated_cifs, no_geom_opts, no_materials = result_queue.get()
        all_cifs.extend(generated_cifs)
        all_no_geom_opts.extend(no_geom_opts)
        all_no_materials.extend(no_materials)

    print(f"CIFs generated: {len(all_cifs):,}")

    with gzip.open(out_file, 'wb') as f:
        pickle.dump(all_cifs, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(no_geom_opt_out_file, "wt") as f:
        f.writelines([f"{o}\n" for o in all_no_geom_opts])

    with open(no_material_out_file, "wt") as f:
        f.writelines([f"{o}\n" for o in all_no_materials])
