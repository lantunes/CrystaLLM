import os.path
from os import listdir
from os.path import isfile, join
import json
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
import numpy as np
import gzip

from lib import extract_space_group_symbol


if __name__ == '__main__':
    src_dir = "../out/nomad_entries_data_2023_04_30"
    out_file = "../out/nomad_2023_04_30.cif.pkl.gz"

    files = [f for f in listdir(src_dir) if isfile(join(src_dir, f))]

    """
    in resp1.json:
    922 successes 
     - symprec=0.1: space group match in 882
     - symprec=0.01: space group match in 869
     - symprec=0.001: space group match in 850 
     - symprec=0.0001: space group match in 842
    75 no geom opt present, 3 no material present
    """
    for json_fname in files:
        with open(os.path.join(src_dir, json_fname), "rt") as f:
            obj = json.load(f)

        for entry in obj["data"]:

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

                    cif_writer = CifWriter(structure, symprec=0.1)
                    cif = comment + str(cif_writer)

                    cif_sg = extract_space_group_symbol(cif)
                    if cif_sg == space_group_symbol:
                        print("SG match\n")

                    print(cif)

                    print("= = = = = = = = = = = = = = = = = = = = = = = = = = = = =")
                else:
                    print("*** No geom opt ***")
            else:
                print("*** No material ***")