import numpy as np
from pymatgen.analysis.local_env import CrystalNN
import warnings

from pymatgen.core import Composition, Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


def abs_r_score(actual, predicted):
    """
    An example comparison between |R| and R^2:
    ```
    import matplotlib.pyplot as plt
    from sklearn.metrics import r2_score

    actual =    np.array([1, 4, 1, 3, 6, 4, 5, 1, 2, 5])
    predicted = np.array([1, 3, 1, 2, 1, 4, 4, 1, 3, 5])

    print(abs_r_score(actual, predicted))
    print(r2_score(actual, predicted))

    plt.scatter(predicted, actual)
    plt.yticks(list(range(7)))
    plt.xticks(list(range(7)))
    plt.show()
    ```
    """
    actual = np.array(actual)
    predicted = np.array(predicted)

    if len(predicted) < 2:
        msg = "|R| score is not well-defined with less than two samples."
        warnings.warn(msg, UserWarning)
        return float("nan")

    # sum of the absolute errors
    sae = np.sum(np.abs(actual - predicted))

    # sum of the absolute deviations from the mean
    sad = np.sum(np.abs(actual - np.mean(actual)))

    return 1 - (sae / sad)


def bond_length_reasonableness_score(structure, tolerance=0.3):
    """
    If a bond length is 30% shorter or longer than the sum of the atomic radii, the score is lower.

    :param structure:
    :return:
    """
    crystal_nn = CrystalNN()

    min_ratio = 1 - tolerance
    max_ratio = 1 + tolerance

    # calculate the score based on bond lengths and covalent radii
    score = 0
    bond_count = 0
    for i, site in enumerate(structure):
        bonded_sites = crystal_nn.get_nn_info(structure, i)
        for connected_site_info in bonded_sites:
            connected_site = connected_site_info['site']
            bond_length = site.distance(connected_site)

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

            # penalize bond lengths that are too short or too long
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
