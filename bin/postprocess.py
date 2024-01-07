import os
import argparse

from crystallm import (
    extract_space_group_symbol,
    replace_symmetry_operators,
    remove_atom_props_block,
)


def postprocess(cif: str) -> str:
    # replace the symmetry operators with the correct operators
    space_group_symbol = extract_space_group_symbol(cif)
    if space_group_symbol is not None and space_group_symbol != "P 1":
        cif = replace_symmetry_operators(cif, space_group_symbol)

    # remove atom props
    cif = remove_atom_props_block(cif)

    return cif


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Post-process CIF files.")
    parser.add_argument("name", type=str,
                        help="Path to the directory containing the raw CIF files to be post-processed.")
    parser.add_argument("out", type=str, required=True,
                        help="Path to the directory where the post-processed CIF files should be written")

    args = parser.parse_args()

    input_dir = args.name
    output_dir = args.out

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".cif"):
            input_path = os.path.join(input_dir, filename)
            with open(input_path, "r") as file:
                cif_str = file.read()
                processed_cif = postprocess(cif_str)

            output_path = os.path.join(output_dir, filename)
            with open(output_path, "w") as file:
                file.write(processed_cif)
            print(f"processed: {filename}")
