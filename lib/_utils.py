import os
import re

THIS_DIR = os.path.dirname(os.path.abspath(__file__))


with open(os.path.join(THIS_DIR, "spacegroups.txt"), "rt") as f:
    SPACE_GROUPS = [sg.strip() for sg in f.readlines()]


ATOMS = ["Si", "C", "Pb", "I", "Br", "Cl", "Eu", "O", "Fe", "Sb", "In", "S", "N", "U", "Mn", "Lu", "Se", "Tl", "Hf",
         "Ir", "Ca", "Ta", "Cr", "K", "Pm", "Mg", "Zn", "Cu", "Sn", "Ti", "B", "W", "P", "H", "Pd", "As", "Co", "Np",
         "Tc", "Hg", "Pu", "Al", "Tm", "Tb", "Ho", "Nb", "Ge", "Zr", "Cd", "V", "Sr", "Ni", "Rh", "Th", "Na", "Ru",
         "La", "Re", "Y", "Er", "Ce", "Pt", "Ga", "Li", "Cs", "F", "Ba", "Te", "Mo", "Gd", "Pr", "Bi", "Sc", "Ag", "Rb",
         "Dy", "Yb", "Nd", "Au", "Os", "Pa", "Sm", "Be", "Ac", "Xe", "Kr", "He", "Ne", "Ar"]

DIGITS = [str(d) for d in list(range(10))]

KEYWORDS = [
    "_cell_length_b",
    "_atom_site_occupancy",
    "_atom_site_attached_hydrogens",
    "_cell_length_a",
    "_cell_angle_beta",
    "_symmetry_equiv_pos_as_xyz",
    "_cell_angle_gamma",
    "_atom_site_fract_x",
    "_symmetry_space_group_name_H-M",
    "_symmetry_Int_Tables_number",
    "_chemical_formula_structural",
    "_chemical_name_systematic",
    "_atom_site_fract_y",
    "_atom_site_symmetry_multiplicity",
    "_chemical_formula_sum",
    "_atom_site_label",
    "_atom_site_type_symbol",
    "_cell_length_c",
    "_atom_site_B_iso_or_equiv",
    "_symmetry_equiv_pos_site_id",
    "_cell_volume",
    "_atom_site_fract_z",
    "_cell_angle_alpha",
    "_cell_formula_units_Z",
    "loop_",
    "data_"
]

SYMBOLS = [
    "x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"
]

TOKENS = list(ATOMS)
TOKENS.extend(DIGITS)
TOKENS.extend(KEYWORDS)
TOKENS.extend(SYMBOLS)
TOKENS.extend(SPACE_GROUPS)

UNK_TOKEN = "<unk>"

ESCAPED_TOKENS = [re.escape(token) for token in TOKENS]
ESCAPED_TOKENS.sort(key=len, reverse=True)


def tokenize_cif(cif_string, single_spaces=True):
    # Create a regex pattern by joining the escaped tokens with '|'
    token_pattern = '|'.join(ESCAPED_TOKENS)

    # Add a regex pattern to match any sequence of characters separated by whitespace or punctuation
    full_pattern = f'({token_pattern}|\\w+|[\\.,;!?])'

    # Tokenize the input string using the regex pattern
    if single_spaces:
        cif_string = re.sub(r'[ \t]+', ' ', cif_string)
    tokens = re.findall(full_pattern, cif_string)

    # Replace unrecognized tokens with the unknown_token
    tokens = [token if token in TOKENS else UNK_TOKEN for token in tokens]

    return tokens


TOKENS_WITH_UNK = list(TOKENS)
TOKENS_WITH_UNK.append(UNK_TOKEN)

# a mapping from characters to integers
TOKEN_TO_ID = {ch: i for i, ch in enumerate(TOKENS_WITH_UNK)}
ID_TO_TOKEN = {i: ch for i, ch in enumerate(TOKENS_WITH_UNK)}


def encode(s):
    # encoder: take a string, output a list of integers
    return [TOKEN_TO_ID[c] for c in s]


def decode(l):
    # decoder: take a list of integers, output a string
    return ''.join([ID_TO_TOKEN[i] for i in l])
