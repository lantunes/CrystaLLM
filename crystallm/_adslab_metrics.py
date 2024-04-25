from pymatgen.io.cif import CifParser
from pymatgen.analysis.local_env import CrystalNN
from pymatgen.core.structure import Molecule, Structure, Composition, Element
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from ase.io import read
import os, re
import networkx as nx


def check_indices_validity(structure, bulk_indices, adsorbate_indices) -> bool:
    """ Validates if the provided indices are within the bounds of the structure and non-overlapping. """
    structure_length = len(structure)
    if not all(0 <= i < structure_length for i in bulk_indices + adsorbate_indices):
        raise ValueError("Some indices are out of the valid range for the structure.")

    if set(bulk_indices).intersection(adsorbate_indices):
        raise ValueError("Overlap found between bulk and adsorbate indices.")

    return True


def segregate_structure(adslab_structure, bulk_symbols, ads_symbols) -> tuple:
    """
    Separates the indices of atoms in a structure into bulk and adsorbate based on their symbols.
    """
    bulk_composition = Composition(bulk_symbols)
    ads_composition = Composition(ads_symbols)

    adsorbate_indices = []
    bulk_indices = []

    # Initialize lists to hold indices of adsorbate and bulk atoms
    total_composition = adslab_structure.composition.as_dict()
    for element, amt in total_composition.items():
        if element in bulk_composition and element not in ads_composition:
            bulk_indices += [i for i, site in enumerate(adslab_structure) if site.specie.symbol == element]
        elif element in ads_composition and element not in bulk_composition:

            ads_indices = [i for i, site in enumerate(adslab_structure) if site.specie.symbol == element]
            adsorbate_indices += ads_indices

    return bulk_indices, adsorbate_indices


# def segregate_structure(cif, bulk_symbols, ads_symbols):
#     """ Separates bulk and adsorbate indices based on their symbols. """
#     parser = CifParser.from_str(cif)
#     structure = parser.get_structures()[0]
#     bulk_indices = [idx for idx, site in enumerate(structure) if site.specie.symbol in bulk_symbols]
#     adsorbate_indices = [idx for idx, site in enumerate(structure) if site.specie.symbol in ads_symbols]
    
#     check_indices_validity(structure, bulk_indices, adsorbate_indices)
    
#     bulk_structure = structure[bulk_indices]
#     adsorbate_structure = structure[adsorbate_indices]
    
#     return bulk_indices, bulk_structure, adsorbate_indices, adsorbate_structure



# def segregate_structure(cif: str, bulk_symbols: list[str], ads_symbols: list[str]) -> tuple:
#     """
#     Separates bulk and adsorbate indices and structures based on their atomic symbols from CIF data.
    
#     Args:
#     cif (str): The CIF file content as a string.
#     bulk_symbols (list[str]): List of symbols representing the bulk atoms.
#     ads_symbols (list[str]): List of symbols representing the adsorbate atoms.

#     Returns:
#     tuple: A tuple containing:
#            - bulk_indices (list[int]): Indices of the bulk atoms.
#            - bulk_structure (Structure): The bulk part of the structure.
#            - adsorbate_indices (list[int]): Indices of the adsorbate atoms.
#            - adsorbate_structure (Structure): The adsorbate part of the structure.
#     """
#     parser = CifParser.from_str(cif)
#     structure = parser.get_structures()[0]

#     bulk_indices = [idx for idx, site in enumerate(structure) if site.specie.symbol in bulk_symbols]
#     adsorbate_indices = [idx for idx, site in enumerate(structure) if site.specie.symbol in ads_symbols]

#     check_indices_validity(structure, bulk_indices, adsorbate_indices)

#     # bulk_structure = structure.copy()
#     # bulk_structure.remove_sites([i for i in range(len(structure)) if i not in bulk_indices])
    
#     # adsorbate_structure = structure.copy()
#     # adsorbate_structure.remove_sites([i for i in range(len(structure)) if i not in adsorbate_indices])

#     return bulk_indices, adsorbate_indices


def is_molecule(adslab_structure, adsorbate_indices, cutoff_multiplier=1.3) -> bool:
    """ Checks if the subset defined by adsorbate_indices forms a valid molecule. """
    adsorbate = Molecule([adslab_structure[i].specie for i in adsorbate_indices],
                         [adslab_structure[i].coords for i in adsorbate_indices])
    # breakpoint()
    #return adsorbate.is_ordered and adsorbate.is_valid()
    if not (adsorbate.is_ordered and adsorbate.is_valid()):
        return False
    # Building the graph from the molecule
    G = nx.Graph()
    G.add_nodes_from(range(len(adsorbate_indices)))
    # breakpoint()
    # Add edges based on the sum of covalent radii of the atoms
    for i in range(len(adsorbate)):

        for j in range(i + 1, len(adsorbate)):

            dist = adsorbate.get_distance(i, j)
            covalent_dist = adsorbate[i].specie.atomic_radius + adsorbate[j].specie.atomic_radius

            if dist <= covalent_dist*cutoff_multiplier:
                G.add_edge(i, j)
    # breakpoint()
    return nx.is_connected(G)

def is_fully_connected(adslab_structure, bulk_indices) -> bool:
    """
    Verifies if all sites in the structure that belong to the bulk (slab) are fully connected
    using nearest neighbors. This function only checks connectivity for the specified bulk_indices.
    """
    cnn = CrystalNN()
    neighbor_info = {index: cnn.get_nn_info(adslab_structure, index) for index in bulk_indices}

    # Check if every site in bulk_indices is connected to at least one other site in bulk_indices
    return all(
        any(neighbor['site_index'] in bulk_indices for neighbor in neighbor_info[index])
        for index in bulk_indices
    )
    
    # # Filter neighbor info to only include connections between bulk indices
    # return all(
    #     any(neighbor['site_index'] in bulk_indices for neighbor in cnn.get_nn_info(adslab_structure, index))
    #     for index in bulk_indices
    # )

def is_adsorbed(adslab_structure, adsorbate_indices) -> bool:
    """ Determines if any adsorbate atom has a neighbor from the slab. """
    nn_finder = CrystalNN()
    for i in adsorbate_indices:
        neighbors = nn_finder.get_nn_info(adslab_structure, i)
        if any(neighbor['site_index'] not in adsorbate_indices for neighbor in neighbors):
            return True
    return False


def is_bond_length_reasonable(adslab_structure, subset_indices, tolerance=0.3, h_factor=2.5):
    """
    If a bond length is 30% shorter or longer than the sum of the atomic radii, the score is lower.
    """
    structure = adslab_structure
    structure = Structure.from_sites([structure[i] for i in subset_indices])
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


def validate_ads_slab(cif_content, bulk_symbols, ads_symbols) -> str:
    """ Validates the slab and adsorbate configuration from a CIF content string. """
    parser = CifParser.from_string(cif_content)
    adslab_structure = parser.get_structures()[0]
    
    bulk_indices, adsorbate_indices = segregate_structure(adslab_structure, bulk_symbols, ads_symbols)
    
    if not is_fully_connected(adslab_structure):
        return "Slab part is not fully connected."
    if not is_molecule(adslab_structure, adsorbate_indices):
        return "Adsorbate part does not form a proper molecule."
    if not is_adsorbed(adslab_structure, adsorbate_indices):
        return "Adsorbate is not properly adsorbed on the slab surface."

    return "Validation successful: Adsorbate is properly adsorbed on the slab."


def match_adsorbate_composition(adslab_structure, adsorbate_indices, expected_formula) -> bool:
    """
    Checks if the composition of the adsorbate in an adslab structure matches a given chemical formula.
    """
    # Extract adsorbate sites based on provided indices
    adsorbate_sites = [adslab_structure[i] for i in adsorbate_indices]

    # Calculate the composition of these adsorbate sites
    adsorbate_composition = Composition()
    for site in adsorbate_sites:
        element_string = str(site.specie)
        adsorbate_composition += Composition(element_string)

    # Normalize and compare the adsorbate composition to the expected formula
    adsorbate_composition_reduced = adsorbate_composition.get_reduced_formula_and_factor()[0]
    expected_composition = Composition(expected_formula)
    expected_composition_reduced = expected_composition.get_reduced_formula_and_factor()[0]

    # Check if the normalized formulas match
    match = adsorbate_composition_reduced == expected_composition_reduced

    return match, adsorbate_composition_reduced, expected_composition_reduced

def match_slab_composition_ratio(adslab_structure, bulk_indices, expected_formula) -> bool:
    """
    Evaluates if the composition ratio of the slab in an adslab structure matches the expected ratios.
    """
    # Extract slab sites based on provided indices
    slab_sites = [adslab_structure[i] for i in bulk_indices]
    
    # Sum up the compositions of individual slab sites
    slab_composition = Composition()
    for site in slab_sites:
        # Convert site species to a string that Composition can handle
        element_string = str(site.specie)
        slab_composition += Composition(element_string)
    
    # Get the reduced composition directly as a Composition object
    slab_composition_reduced, _ = slab_composition.get_reduced_composition_and_factor()

    # Construct and reduce the expected composition directly
    expected_composition = Composition(expected_formula)
    expected_composition_reduced, _ = expected_composition.get_reduced_composition_and_factor()

    # Compare the reduced compositions
    match = slab_composition_reduced.almost_equals(expected_composition_reduced)

    return match, slab_composition_reduced, expected_composition_reduced



# def load_labels(sid, directory):
#     # preprocess the system id
#     sid = re.match(r'random\d+', sid).group()
#     # for directory in directories:
#     label_file = os.path.join(directory, f'{sid}.traj')
#     if os.path.exists(label_file):
#         atoms = read(labsel_file, '-1')
#         structure = AseAtomsAdaptor.get_structure(atoms)
#         return structure, atoms.get_potential_energy()
#     raise FileNotFoundError(f'Trajectory file for system ID {sid} not found in any of the specified directories.')


def evaluate_structure_similarity(cif_str, label_structure):
    parser = CifParser.from_str(cif_str)
    gen_structure = parser.get_structures()[0]
    sm = StructureMatcher()
    if sm.fit(gen_structure, label_structure):
        # rmsd = sm.get_rmsd(gen_structure, label_structure)
        rmsd = sm.get_rms_dist()
        return rmsd
    else:
        return float('inf')
    

def load_and_evaluate_similarity(sid, directory, gen_structure):
    """
    Loads a structure from a file, evaluates its potential energy, and compares its similarity to another structure provided as a CIF string.
    """
    # Preprocess the system id
    sid = re.match(r'random\d+', sid).group()

    # Load the structure from the specified directory
    label_file = os.path.join(directory, f'{sid}.traj')
    if os.path.exists(label_file):
        atoms = read(label_file, '-1')
        label_structure = AseAtomsAdaptor.get_structure(atoms)
    else:
        raise FileNotFoundError(f'Trajectory file for system ID {sid} not found in the specified directory: {directory}')

    # Compare the structures
    sm = StructureMatcher()
    if sm.fit(label_structure, gen_structure):
        rmsd = sm.get_rms_dist(label_structure, gen_structure) #sm.get_rmsd()
    else:
        rmsd = float('inf')  # Use infinity to indicate no match

    return rmsd