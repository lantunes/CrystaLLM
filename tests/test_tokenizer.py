import unittest
import inspect
from crystallm import CIFTokenizer


class TestSomething(unittest.TestCase):

    def test_tokenize_cif(self):
        tokenizer = CIFTokenizer()

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

    def test_tokenize_cif_atoms_like_spacegroup(self):
        tokenizer = CIFTokenizer()

        cif_str = inspect.cleandoc('''
        data_Na1P1
        loop_
        _atom_type_symbol
        _atom_type_electronegativity
        Na 1.1300
        P 1.8800
        _symmetry_space_group_name_H-M    P1
        _chemical_formula_structural  NaP
        _chemical_formula_sum 'Na1 P1'
        loop_
        _atom_site_type_symbol
        _atom_site_label
        _atom_site_symmetry_multiplicity
        Na Na0 1
        P P1 1
        ''')

        tokens = tokenizer.tokenize_cif(cif_str)

        assert tokens == [
            'data_', 'Na', '1', 'P', '1', '\n',
            'loop_', '\n',
            '_atom_type_symbol', '\n',
            '_atom_type_electronegativity', '\n',
            'Na', ' ', '1', '.', '1', '3', '0', '0', '\n',
            'P', ' ', '1', '.', '8', '8', '0', '0', '\n',
            '_symmetry_space_group_name_H-M', ' ', 'P1_sg', '\n',
            '_chemical_formula_structural', ' ', 'Na', 'P', '\n',
            '_chemical_formula_sum', ' ', "'", 'Na', '1', ' ', 'P', '1', "'", '\n',
            'loop_', '\n',
            '_atom_site_type_symbol', '\n',
            '_atom_site_label', '\n',
            '_atom_site_symmetry_multiplicity', '\n',
            'Na', ' ', 'Na', '0', ' ', '1', '\n',
            'P', ' ', 'P', '1', ' ', '1'
        ]

    def test_tokenize_cif_space_group(self):
        tokenizer = CIFTokenizer()

        cif_str = "_symmetry_space_group_name_H-M    Pm-3m\n"
        tokens = tokenizer.tokenize_cif(cif_str)
        assert tokens == ['_symmetry_space_group_name_H-M', ' ', 'Pm-3m_sg', '\n']

        cif_str = "_symmetry_space_group_name_H-M Pm-3m\n"
        tokens = tokenizer.tokenize_cif(cif_str)
        assert tokens == ['_symmetry_space_group_name_H-M', ' ', 'Pm-3m_sg', '\n']

        cif_str = "_symmetry_space_group_name_H-M Pmn2_1\n"
        tokens = tokenizer.tokenize_cif(cif_str)
        assert tokens == ['_symmetry_space_group_name_H-M', ' ', 'Pmn2_1_sg', '\n']

        cif_str = "_symmetry_space_group_name_H-M I4/m\n"
        tokens = tokenizer.tokenize_cif(cif_str)
        assert tokens == ['_symmetry_space_group_name_H-M', ' ', 'I4/m_sg', '\n']

        cif_str = "_symmetry_space_group_name_H-M P1\n"
        tokens = tokenizer.tokenize_cif(cif_str)
        assert tokens == ['_symmetry_space_group_name_H-M', ' ', 'P1_sg', '\n']

    def test_encode_decode(self):
        tokenizer = CIFTokenizer()

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

    def test_encode_decode_atoms_like_spacegroup(self):
        tokenizer = CIFTokenizer()

        cif_str = inspect.cleandoc('''
        data_Na1P1
        loop_
        _atom_type_symbol
        _atom_type_electronegativity
        Na 1.1300
        P 1.8800
        _symmetry_space_group_name_H-M P1
        _chemical_formula_structural NaP
        _chemical_formula_sum 'Na1 P1'
        loop_
        _atom_site_type_symbol
        _atom_site_label
        _atom_site_symmetry_multiplicity
        Na Na0 1
        P P1 1
        ''')

        tokens = tokenizer.tokenize_cif(cif_str)
        encoded = tokenizer.encode(tokens)

        assert len(encoded) == len(tokens)

        for i, id in enumerate(encoded):
            if tokens[i] == "P1_sg":
                assert tokenizer.id_to_token[id] == "P1"
            else:
                assert tokenizer.id_to_token[id] == tokens[i]

        decoded = tokenizer.decode(encoded)

        assert "".join(decoded) == cif_str
