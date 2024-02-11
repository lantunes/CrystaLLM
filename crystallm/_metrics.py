import re

from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from ._utils import extract_data_formula


def bond_length_reasonableness_score(cif_str, tolerance=0.32, h_factor=2.5):
    """
    If a bond length is 30% shorter or longer than the sum of the atomic radii, the score is lower.
    """
    structure = Structure.from_str(cif_str, fmt="cif")
    crystal_nn = CrystalNN()

    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance

    # calculate the score based on bond lengths and covalent radii
    score = 0
    bond_count = 0
    for i, site in enumerate(structure):
        bonded_sites = crystal_nn.get_nn_info(structure, i)
        for connected_site_info in bonded_sites:
            j = connected_site_info['site_index']
            if i == j:  # skip if they're the same site
                continue
            connected_site = connected_site_info['site']
            bond_length = site.distance(connected_site)

            is_hydrogen_bond = "H" in [site.specie.symbol, connected_site.specie.symbol]

            electronegativity_diff = abs(site.specie.X - connected_site.specie.X)
            """
            According to the Pauling scale, when the electronegativity difference 
            between two bonded atoms is less than 1.7, the bond can be considered 
            to have predominantly covalent character, while a difference greater 
            than or equal to 1.7 indicates that the bond has significant ionic 
            character.
            """
            if electronegativity_diff >= 1.7:
                # use ionic radii
                if site.specie.X < connected_site.specie.X:
                    expected_length = site.specie.average_cationic_radius + connected_site.specie.average_anionic_radius
                else:
                    expected_length = site.specie.average_anionic_radius + connected_site.specie.average_cationic_radius
            else:
                expected_length = site.specie.atomic_radius + connected_site.specie.atomic_radius

            bond_ratio = bond_length / expected_length

            # penalize bond lengths that are too short or too long;
            #  check if bond involves hydrogen and adjust tolerance accordingly
            if is_hydrogen_bond:
                if bond_ratio < h_factor:
                    score += 1
            else:
                if min_ratio < bond_ratio < max_ratio:
                    score += 1

            bond_count += 1

    normalized_score = score / bond_count

    return normalized_score


def is_space_group_consistent(cif_str):
    structure = Structure.from_str(cif_str, fmt="cif")
    parser = CifParser.from_string(cif_str)
    cif_data = parser.as_dict()

    # Extract the stated space group from the CIF file
    stated_space_group = cif_data[list(cif_data.keys())[0]]['_symmetry_space_group_name_H-M']

    # Analyze the symmetry of the structure
    spacegroup_analyzer = SpacegroupAnalyzer(structure, symprec=0.1)

    # Get the detected space group
    detected_space_group = spacegroup_analyzer.get_space_group_symbol()

    # Check if the detected space group matches the stated space group
    is_match = stated_space_group.strip() == detected_space_group.strip()

    return is_match


def is_formula_consistent(cif_str):
    parser = CifParser.from_string(cif_str)
    cif_data = parser.as_dict()

    formula_data = Composition(extract_data_formula(cif_str))
    formula_sum = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"])
    formula_structural = Composition(cif_data[list(cif_data.keys())[0]]["_chemical_formula_structural"])

    return formula_data.reduced_formula == formula_sum.reduced_formula == formula_structural.reduced_formula


def is_atom_site_multiplicity_consistent(cif_str):
    # Parse the CIF string
    parser = CifParser.from_string(cif_str)
    cif_data = parser.as_dict()

    # Extract the chemical formula sum from the CIF data
    formula_sum = cif_data[list(cif_data.keys())[0]]["_chemical_formula_sum"]

    # Convert the formula sum into a dictionary
    expected_atoms = Composition(formula_sum).as_dict()

    # Count the atoms provided in the _atom_site_type_symbol section
    actual_atoms = {}
    for key in cif_data:
        if "_atom_site_type_symbol" in cif_data[key] and "_atom_site_symmetry_multiplicity" in cif_data[key]:
            for atom_type, multiplicity in zip(cif_data[key]["_atom_site_type_symbol"],
                                               cif_data[key]["_atom_site_symmetry_multiplicity"]):
                if atom_type in actual_atoms:
                    actual_atoms[atom_type] += int(multiplicity)
                else:
                    actual_atoms[atom_type] = int(multiplicity)

    # Validate if the expected and actual atom counts match
    return expected_atoms == actual_atoms


def is_sensible(cif_str, length_lo=0.5, length_hi=1000., angle_lo=10., angle_hi=170.):
    cell_length_pattern = re.compile(r"_cell_length_[abc]\s+([\d\.]+)")
    cell_angle_pattern = re.compile(r"_cell_angle_(alpha|beta|gamma)\s+([\d\.]+)")

    cell_lengths = cell_length_pattern.findall(cif_str)
    for length_str in cell_lengths:
        length = float(length_str)
        if length < length_lo or length > length_hi:
            return False

    cell_angles = cell_angle_pattern.findall(cif_str)
    for _, angle_str in cell_angles:
        angle = float(angle_str)
        if angle < angle_lo or angle > angle_hi:
            return False

    return True


def is_valid(cif_str, bond_length_acceptability_cutoff=1.0):
    if not is_formula_consistent(cif_str):
        return False
    if not is_atom_site_multiplicity_consistent(cif_str):
        return False
    bond_length_score = bond_length_reasonableness_score(cif_str)
    if bond_length_score < bond_length_acceptability_cutoff:
        return False
    if not is_space_group_consistent(cif_str):
        return False
    return True

