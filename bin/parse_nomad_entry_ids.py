from os import listdir
from os.path import isfile, join
import json
import gzip
import csv
from tqdm import tqdm

import warnings

warnings.filterwarnings(action="ignore", category=UserWarning)


if __name__ == '__main__':
    src_dir = "../out/nomad_entries"
    out_file = "../out/nomad_entries_2023_04_30.csv.gz"

    files = [f for f in listdir(src_dir) if isfile(join(src_dir, f))]

    rows = []
    columns = [
        "material_id",
        "chemical_formula_descriptive",
        "chemical_formula_reduced",
        "elements_exclusive",
        "entry_id"
    ]

    for fname in tqdm(files):
        with open(f"{src_dir}/{fname}", "rt") as f:
            result = json.load(f)

            for material in result["data"]:

                # try to use the final energy difference to determine which entry to use
                lowest_energy = float("inf")
                lowest_energy_entry_id = None
                for entry in material["entries"]:
                    if "results" in entry and "properties" in entry["results"] and \
                            "geometry_optimization" in entry["results"]["properties"] and \
                            "final_energy_difference" in entry["results"]["properties"]["geometry_optimization"]:
                        energy = float(entry["results"]["properties"]["geometry_optimization"]["final_energy_difference"])
                        if lowest_energy_entry_id is None or energy < lowest_energy:
                            lowest_energy = energy
                            lowest_energy_entry_id = entry["entry_id"]

                # use the first entry if still undecided
                if lowest_energy_entry_id is None:
                    lowest_energy_entry_id = material["entries"][0]["entry_id"]

                rows.append([
                    material["material_id"],
                    material["chemical_formula_descriptive"],
                    material["chemical_formula_reduced"],
                    material["elements_exclusive"],
                    lowest_energy_entry_id,
                ])

    with gzip.open(out_file, "wt") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        for row in rows:
            writer.writerow(row)
