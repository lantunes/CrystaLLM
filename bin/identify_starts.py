import os
import tarfile
import pickle
import numpy as np
import argparse
from tqdm import tqdm

from crystallm import (
    extract_data_formula,
    extract_space_group_symbol,
)


def get_underrepresented_set(underrepresented_fname):
    with open(underrepresented_fname, "rb") as f:
        comps = pickle.load(f)
    underrepresented_set = set()
    for comp, sg in comps:
        underrepresented_set.add(f"{comp}_{sg}")
    return underrepresented_set


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Identify start token indices.")
    parser.add_argument("--dataset_fname", type=str, required=True,
                        help="Path to the tokenized dataset file (.tar.gz).")
    parser.add_argument("--out_fname", type=str, required=True,
                        help="Path to the file that will contain the serialized Python list of start indices. "
                             "Recommended extension is `.pkl`.")
    parser.add_argument("--underrepresented_fname", type=str, default=None,
                        help="Optional: Path to the file containing underrepresented sample information. "
                             "The file should be .pkl file with a serialized Python list of "
                             "(cell composition, space group) pairs that are under-represented.")
    parser.add_argument("--underrepresented_out_fname", type=str,
                        help="Optional: Path to the file that will contain the under-represented start indices as a "
                             "serialized Python list. Recommended extension is `.pkl`.")
    args = parser.parse_args()

    dataset_fname = args.dataset_fname
    out_fname = args.out_fname
    underrepresented_fname = args.underrepresented_fname
    underrepresented_out_fname = args.underrepresented_out_fname

    base_path = os.path.splitext(os.path.basename(dataset_fname))[0]
    base_path = os.path.splitext(base_path)[0]

    with tarfile.open(dataset_fname, "r:gz") as file:
        file_content_byte = file.extractfile(f"{base_path}/meta.pkl").read()
        meta = pickle.loads(file_content_byte)

        extracted = file.extractfile(f"{base_path}/train.bin")
        train_ids = np.frombuffer(extracted.read(), dtype=np.uint16)

    underrepresented_set = None
    if underrepresented_fname:
        underrepresented_set = get_underrepresented_set(underrepresented_fname)

    all_cif_start_indices = []
    underrepresented_start_indices = []

    curr_cif_tokens = []

    for i, id in tqdm(enumerate(train_ids), total=len(train_ids), desc="identifying starts..."):
        token = meta["itos"][id]

        if token == "data_":
            all_cif_start_indices.append(i)

        curr_cif_tokens.append(token)

        if len(curr_cif_tokens) > 1 and curr_cif_tokens[-2:] == ['\n', '\n']:
            # we're at the end of the CIF;
            #  reconstruct the CIF and see if it's in the underrepresented set
            cif = ''.join(curr_cif_tokens)
            data_formula = extract_data_formula(cif)
            space_group_symbol = extract_space_group_symbol(cif)

            if underrepresented_set and f"{data_formula}_{space_group_symbol}" in underrepresented_set:
                # the last added start index is the start index of this CIF
                underrepresented_start_indices.append(all_cif_start_indices[-1])

            curr_cif_tokens = []

    print("writing start indices...")
    with open(out_fname, "wb") as f:
        pickle.dump(all_cif_start_indices, f, protocol=pickle.HIGHEST_PROTOCOL)

    if underrepresented_fname and underrepresented_out_fname:
        with open(underrepresented_out_fname, "wb") as f:
            pickle.dump(underrepresented_start_indices, f, protocol=pickle.HIGHEST_PROTOCOL)
