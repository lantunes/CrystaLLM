import csv
import gzip
from tqdm import tqdm
import sentencepiece as spm


if __name__ == '__main__':
    fname = "data/all_cif_structures.csv.gz"

    cif_lines = []

    with gzip.open(fname, "rt") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in tqdm(reader):
            mpid = row[0]
            cif = row[1].replace('"', '')

            lines = cif.split('\\n')
            for line in lines:
                line = line.strip()
                if len(line) > 0 and not line.startswith("#"):
                    cif_lines.append(line)

    keywords = set()

    for cif_line in tqdm(cif_lines):
        tokens = cif_line.split(" ")
        for t in tokens:
            if t.startswith("_"):
                keywords.add(t)

    for kw in keywords:
        print(kw)

    # data_
    # loop_
    print(len(keywords))
