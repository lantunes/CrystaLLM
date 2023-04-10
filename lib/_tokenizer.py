import os
import re
from abc import abstractmethod

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

EXTENDED_KEYWORDS = [
    "_atom_type_symbol",
    "_atom_type_electronegativity",  # does not exist in CIF spec
    "_atom_type_radius",  # does not exist in CIF spec
    "_atom_type_ionic_radius",  # does not exist in CIF spec
    "_atom_type_oxidation_number"
]

UNK_TOKEN = "<unk>"


class CIFTokenizer:
    def __init__(self):
        self._tokens = list(self.atoms())
        self._tokens.extend(self.digits())
        self._tokens.extend(self.keywords())
        self._tokens.extend(self.symbols())
        self._tokens.extend(self.space_groups())

        self._escaped_tokens = [re.escape(token) for token in self._tokens]
        self._escaped_tokens.sort(key=len, reverse=True)

        self._tokens_with_unk = list(self._tokens)
        self._tokens_with_unk.append(UNK_TOKEN)

        # a mapping from characters to integers
        self._token_to_id = {ch: i for i, ch in enumerate(self._tokens_with_unk)}
        self._id_to_token = {i: ch for i, ch in enumerate(self._tokens_with_unk)}

    @abstractmethod
    def atoms(self):
        pass

    @abstractmethod
    def digits(self):
        pass

    @abstractmethod
    def keywords(self):
        pass

    @abstractmethod
    def symbols(self):
        pass

    @abstractmethod
    def space_groups(self):
        pass

    @property
    def token_to_id(self):
        return dict(self._token_to_id)

    @property
    def id_to_token(self):
        return dict(self._id_to_token)

    def encode(self, s):
        # encoder: take a string, output a list of integers
        return [self._token_to_id[c] for c in s]

    def decode(self, l):
        # decoder: take a list of integers, output a string
        return ''.join([self._id_to_token[i] for i in l])

    def tokenize_cif(self, cif_string, single_spaces=True):
        # Create a regex pattern by joining the escaped tokens with '|'
        token_pattern = '|'.join(self._escaped_tokens)

        # Add a regex pattern to match any sequence of characters separated by whitespace or punctuation
        full_pattern = f'({token_pattern}|\\w+|[\\.,;!?])'

        # Tokenize the input string using the regex pattern
        if single_spaces:
            cif_string = re.sub(r'[ \t]+', ' ', cif_string)
        tokens = re.findall(full_pattern, cif_string)

        # Replace unrecognized tokens with the unknown_token
        tokens = [token if token in self._tokens else UNK_TOKEN for token in tokens]

        return tokens


class CIFSymmTokenizer(CIFTokenizer):
    def __init__(self):
        super().__init__()

    def atoms(self):
        return ATOMS

    def digits(self):
        return DIGITS

    def keywords(self):
        return KEYWORDS

    def symbols(self):
        return ["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"]

    def space_groups(self):
        return SPACE_GROUPS


class CIFNoSymmTokenizer(CIFTokenizer):
    def __init__(self):
        super().__init__()

    def atoms(self):
        return ATOMS

    def digits(self):
        return DIGITS

    def keywords(self):
        return KEYWORDS

    def symbols(self):
        return ["x", "y", "z", ".", "(", ")", "'", ",", " ", "\n"]

    def space_groups(self):
        return []


class CIFSymmPropsTokenizer(CIFTokenizer):
    def __init__(self):
        super().__init__()

    def atoms(self):
        return ATOMS

    def digits(self):
        return DIGITS

    def keywords(self):
        kws = list(KEYWORDS)
        kws.extend(EXTENDED_KEYWORDS)
        return kws

    def symbols(self):
        return ["x", "y", "z", ".", "(", ")", "+", "-", "/", "'", ",", " ", "\n"]

    def space_groups(self):
        return SPACE_GROUPS


def get_cif_tokenizer(symmetrized, includes_props=False):
    if symmetrized:
        if includes_props:
            return CIFSymmPropsTokenizer()
        return CIFSymmTokenizer()
    else:
        return CIFNoSymmTokenizer()
