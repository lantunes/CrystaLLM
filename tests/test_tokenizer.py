import inspect
from lib import get_cif_tokenizer


def test_tokenize_cif():
    tokenizer = get_cif_tokenizer(symmetrized=True, includes_props=True)

    cif_str = inspect.cleandoc('''
    data_Np1Co3
    loop_
    _atom_type_symbol
    _atom_type_electronegativity
    Pm 1.1300
    Co 1.8800
    _symmetry_space_group_name_H-M    Pm
    _chemical_formula_structural  PmCo3
    _chemical_formula_sum 'Pm1 Co3'
    loop_
    _atom_site_type_symbol
    _atom_site_label
    _atom_site_symmetry_multiplicity
    Pm Pm0 1
    Co Co1 3
    ''')

    tokens = tokenizer.tokenize_cif(cif_str)

    assert tokens == [
        'data_', 'Np', '1', 'Co', '3', '\n',
        'loop_', '\n',
        '_atom_type_symbol', '\n',
        '_atom_type_electronegativity', '\n',
        'Pm', ' ', '1', '.', '1', '3', '0', '0', '\n',
        'Co', ' ', '1', '.', '8', '8', '0', '0', '\n',
        '_symmetry_space_group_name_H-M', ' ', 'Pm_sg', '\n',
        '_chemical_formula_structural', ' ', 'Pm', 'Co', '3', '\n',
        '_chemical_formula_sum', ' ', "'", 'Pm', '1', ' ', 'Co', '3', "'", '\n',
        'loop_', '\n',
        '_atom_site_type_symbol', '\n',
        '_atom_site_label', '\n',
        '_atom_site_symmetry_multiplicity', '\n',
        'Pm', ' ', 'Pm', '0', ' ', '1', '\n',
        'Co', ' ', 'Co', '1', ' ', '3'
    ]


def test_tokenize_cif_space_group():
    tokenizer = get_cif_tokenizer(symmetrized=True, includes_props=True)

    cif_str = "_symmetry_space_group_name_H-M    Pm-3m"
    tokens = tokenizer.tokenize_cif(cif_str)
    assert tokens == ['_symmetry_space_group_name_H-M', ' ', 'Pm-3m']

    cif_str = "_symmetry_space_group_name_H-M Pm-3m"
    tokens = tokenizer.tokenize_cif(cif_str)
    assert tokens == ['_symmetry_space_group_name_H-M', ' ', 'Pm-3m']

    cif_str = "_symmetry_space_group_name_H-M Pmn2_1"
    tokens = tokenizer.tokenize_cif(cif_str)
    assert tokens == ['_symmetry_space_group_name_H-M', ' ', 'Pmn2_1']

    cif_str = "_symmetry_space_group_name_H-M I4/m"
    tokens = tokenizer.tokenize_cif(cif_str)
    assert tokens == ['_symmetry_space_group_name_H-M', ' ', 'I4/m']


def test_encode_decode():
    tokenizer = get_cif_tokenizer(symmetrized=True, includes_props=True)

    cif_str = inspect.cleandoc('''
    data_Np1Co3
    loop_
    _atom_type_symbol
    _atom_type_electronegativity
    Pm 1.1300
    Co 1.8800
    _symmetry_space_group_name_H-M Pm
    _chemical_formula_structural PmCo3
    _chemical_formula_sum 'Pm1 Co3'
    loop_
    _atom_site_type_symbol
    _atom_site_label
    _atom_site_symmetry_multiplicity
    Pm Pm0 1
    Co Co1 3
    ''')

    tokens = tokenizer.tokenize_cif(cif_str)
    encoded = tokenizer.encode(tokens)

    assert len(encoded) == len(tokens)

    for i, id in enumerate(encoded):
        if tokens[i] == "Pm_sg":
            assert id != tokenizer.token_to_id["Pm"]
            assert tokenizer.id_to_token[id] == "Pm"
        else:
            assert tokenizer.id_to_token[id] == tokens[i]

    decoded = tokenizer.decode(encoded)

    assert "".join(decoded) == cif_str
