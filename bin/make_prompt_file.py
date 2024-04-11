import argparse
import re

from pymatgen.core import Composition

from crystallm import get_atomic_props_block_for_formula


def get_prompt(comp, sg=None):
    # NOTE: we have to use comp.formula, so that the elements are sorted by electronegativity,
    #  which is what the model saw in training; comp.formula looks something like 'Zn1 Cu1 Te1 Se1',
    #  so we have to strip the spaces
    comp_str = comp.formula.replace(" ", "")
    if sg is not None:
        # construct an input string with the space group
        block = get_atomic_props_block_for_formula(comp_str)
        cif_str = f"data_{comp_str}\n{block}\n_symmetry_space_group_name_H-M {sg}\n"
        # strip out any leading or trailing spaces from the prompt
        cif_str = re.sub(r"^[ \t]+|[ \t]+$", "", cif_str, flags=re.MULTILINE)
        return cif_str
    else:
        return f"data_{comp_str}\n"


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Construct a prompt file from the given composition.")
    parser.add_argument("composition", type=str,
                        help="The desired cell composition. There must be no spaces between elements. "
                             "Note that the cell composition contains the total count of each type in the unit cell."
                             "For example, Na2Cl2 is a cell composition specifying a total of 4 atoms in the "
                             "unit cell.")
    parser.add_argument("prompt_fname", type=str,
                        help="The path to the text file where the prompt will be stored.")
    parser.add_argument("--spacegroup", required=False, type=str,
                        help="The desired space group symbol. e.g. Fd-3m, P4_2/n, etc.")

    args = parser.parse_args()
    composition = args.composition
    prompt_fname = args.prompt_fname
    sg = args.spacegroup

    comp = Composition(composition)
    prompt = get_prompt(comp, sg)

    print(f"writing prompt to {prompt_fname} ...")
    with open(prompt_fname, "wt") as f:
        f.write(prompt)
