import math
import numpy as np
import re
from pymatgen.core import Composition
from pymatgen.io.cif import CifBlock
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.core.operations import SymmOp
from pymatgen.analysis.bond_valence import BVAnalyzer
from pymatgen.transformations.standard_transformations import OxidationStateDecorationTransformation
from itertools import permutations
from sklearn.metrics import mean_absolute_error, r2_score
from ._metrics import abs_r_score


def get_unit_cell_volume(a, b, c, alpha_deg, beta_deg, gamma_deg):
    alpha_rad = math.radians(alpha_deg)
    beta_rad = math.radians(beta_deg)
    gamma_rad = math.radians(gamma_deg)

    volume = (a * b * c * math.sqrt(1 - math.cos(alpha_rad) ** 2 - math.cos(beta_rad) ** 2 - math.cos(gamma_rad) ** 2 +
                                    2 * math.cos(alpha_rad) * math.cos(beta_rad) * math.cos(gamma_rad)))

    return volume


def remove_outliers(actual_values, predicted_values, multiplier=1.5):

    def get_outlier_bounds(values, multiplier=multiplier):
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        return lower_bound, upper_bound

    actual_lower_bound, actual_upper_bound = get_outlier_bounds(actual_values)
    predicted_lower_bound, predicted_upper_bound = get_outlier_bounds(predicted_values)

    filtered_indices = [
        i for i, (actual, predicted) in enumerate(zip(actual_values, predicted_values))
        if actual_lower_bound <= actual <= actual_upper_bound
           and predicted_lower_bound <= predicted <= predicted_upper_bound
    ]

    filtered_actual_values = [actual_values[i] for i in filtered_indices]
    filtered_predicted_values = [predicted_values[i] for i in filtered_indices]

    return filtered_actual_values, filtered_predicted_values


def plot_true_vs_predicted(ax, true_y, predicted_y, xlabel="true", outlier_multiplier=None, ylabel="predicted",
                           min_extra=1, max_extra=1, text=None, metrics=True,
                           alpha=None, title=None, trim_lims=False, size=3, color="lightblue",
                           legend_labels=None, legend_fontsize=6, legend_title=None, legend_loc=None):

    n_outliers_removed = 0

    if outlier_multiplier is not None:
        orig_count = len(true_y)
        true_y, predicted_y = remove_outliers(true_y, predicted_y, outlier_multiplier)
        n_outliers_removed = orig_count - len(true_y)
        print(f"{text}: outliers removed: {n_outliers_removed}/{orig_count} "
              f"({((orig_count - len(true_y))/orig_count)*100:.1f}%)")

    line_start = np.min([np.min(true_y), np.min(predicted_y)]) - min_extra
    line_end = np.max([np.max(true_y), np.max(predicted_y)]) + max_extra

    scatter = ax.scatter(true_y, predicted_y, s=size, linewidth=0.1, edgecolor="black", c=color, alpha=alpha)
    ax.plot([line_start, line_end], [line_start, line_end], 'k-', linewidth=0.35)
    if trim_lims:
        ax.set_xlim(np.min(true_y) - min_extra, np.max(true_y) + max_extra)
        ax.set_ylim(np.min(predicted_y) - min_extra, np.max(predicted_y) + max_extra)
    else:
        ax.set_xlim(line_start, line_end)
        ax.set_ylim(line_start, line_end)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)

    if title is not None:
        ax.set_title(title)

    if text:
        ax.text(0.01, 0.92, text, transform=ax.transAxes)

    if metrics:
        r2 = r2_score(true_y, predicted_y)
        mae = mean_absolute_error(true_y, predicted_y)
        abs_r = abs_r_score(true_y, predicted_y)
        metrics_text = f"$R^2$: {r2:.2f}, MAE: {mae:.4f}, $|R|$: {abs_r:.2f}"
        ax.text(0.2, 0.01, metrics_text, transform=ax.transAxes)

    if legend_labels is not None:
        leg = ax.legend(handles=scatter.legend_elements()[0], labels=legend_labels, fontsize=legend_fontsize,
                         markerscale=0.5, loc=legend_loc)
        if legend_title is not None:
            leg.set_title(legend_title, prop={'size': legend_fontsize})

    return n_outliers_removed


def get_composition_permutations(composition_str):
    composition = Composition(composition_str)
    elements = tuple(composition.elements)
    unique_permutations = set(permutations(elements))

    permuted_compositions = [
        "".join([str(el) + (str(int(composition[el])) if composition[el] != 1 else "") for el in perm]) for perm in
        unique_permutations]

    return permuted_compositions


def get_oxi_state_decorated_structure(structure):
    """
    first tries to use BVAnalyzer, and if that doesn't work (i.e. it raises a ValueError),
    it uses ICSD statistics
    """
    try:
        bva = BVAnalyzer()
        # NOTE: this will raise a ValueError if the oxidation states can't be determined
        struct = bva.get_oxi_state_decorated_structure(structure)
    except ValueError:
        comp = structure.composition
        oxi_transform = OxidationStateDecorationTransformation(
            comp.oxi_state_guesses()[0]
        )
        struct = oxi_transform.apply_transformation(structure)

    return struct


def get_atomic_props_block(composition, oxi=False):
    noble_vdw_radii = {
        "He": 1.40,
        "Ne": 1.54,
        "Ar": 1.88,
        "Kr": 2.02,
        "Xe": 2.16,
        "Rn": 2.20,
    }

    allen_electronegativity = {
        "He": 4.16,
        "Ne": 4.79,
        "Ar": 3.24,
    }

    def _format(val):
        return f"{float(val): .4f}"

    def _format_X(elem):
        if math.isnan(elem.X) and str(elem) in allen_electronegativity:
            return allen_electronegativity[str(elem)]
        return _format(elem.X)

    def _format_radius(elem):
        if elem.atomic_radius is None and str(elem) in noble_vdw_radii:
            return noble_vdw_radii[str(elem)]
        return _format(elem.atomic_radius)

    props = {str(el): (_format_X(el), _format_radius(el), _format(el.average_ionic_radius))
             for el in sorted(composition.elements)}

    data = {}
    data["_atom_type_symbol"] = list(props)
    data["_atom_type_electronegativity"] = [v[0] for v in props.values()]
    data["_atom_type_radius"] = [v[1] for v in props.values()]
    # use the average ionic radius
    data["_atom_type_ionic_radius"] = [v[2] for v in props.values()]

    loop_vals = [
        "_atom_type_symbol",
        "_atom_type_electronegativity",
        "_atom_type_radius",
        "_atom_type_ionic_radius"
    ]

    if oxi:
        symbol_to_oxinum = {str(el): (float(el.oxi_state), _format(el.ionic_radius)) for el in sorted(composition.elements)}
        data["_atom_type_oxidation_number"] = [v[0] for v in symbol_to_oxinum.values()]
        # if we know the oxidation state of the element, use the ionic radius for the given oxidation state
        data["_atom_type_ionic_radius"] = [v[1] for v in symbol_to_oxinum.values()]
        loop_vals.append("_atom_type_oxidation_number")

    loops = [loop_vals]

    return str(CifBlock(data, loops, "")).replace("data_\n", "")


def replace_symmetry_operators(cif_str, space_group_symbol):
    space_group = SpaceGroup(space_group_symbol)
    symmetry_ops = space_group.symmetry_ops

    loops = []
    data = {}
    symmops = []
    for op in symmetry_ops:
        v = op.translation_vector
        symmops.append(SymmOp.from_rotation_and_translation(op.rotation_matrix, v))

    ops = [op.as_xyz_string() for op in symmops]
    data["_symmetry_equiv_pos_site_id"] = [f"{i}" for i in range(1, len(ops) + 1)]
    data["_symmetry_equiv_pos_as_xyz"] = ops

    loops.append(["_symmetry_equiv_pos_site_id", "_symmetry_equiv_pos_as_xyz"])

    symm_block = str(CifBlock(data, loops, "")).replace("data_\n", "")

    pattern = r"(loop_\n_symmetry_equiv_pos_site_id\n_symmetry_equiv_pos_as_xyz\n1 'x, y, z')"
    cif_str_updated = re.sub(pattern, symm_block, cif_str)

    return cif_str_updated


def extract_space_group_symbol(cif_str):
    match = re.search(r"_symmetry_space_group_name_H-M\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract space group from:\n{cif_str}")


def extract_numeric_property(cif_str, prop, numeric_type=float):
    match = re.search(rf"{prop}\s+([.0-9]+)", cif_str)
    if match:
        return numeric_type(match.group(1))
    raise Exception(f"could not find {prop} in:\n{cif_str}")


def extract_volume(cif_str):
    return extract_numeric_property(cif_str, "_cell_volume")


def extract_formula_units(cif_str):
    return extract_numeric_property(cif_str, "_cell_formula_units_Z", numeric_type=int)


def extract_formula_nonreduced(cif_str):
    match = re.search(r"_chemical_formula_sum\s+('([^']+)'|(\S+))", cif_str)
    if match:
        return match.group(2) if match.group(2) else match.group(3)
    raise Exception(f"could not extract _chemical_formula_sum value from:\n{cif_str}")


def semisymmetrize_cif(cif_str):
    return re.sub(
        r"(_symmetry_equiv_pos_as_xyz\n)(.*?)(?=\n(?:\S| \S))",
        r"\1  1  'x, y, z'",
        cif_str,
        flags=re.DOTALL
    )


def replace_data_formula_with_nonreduced_formula(cif_str):
    pattern = r"_chemical_formula_sum\s+(.+)\n"
    pattern_2 = r"(data_)(.*?)(\n)"
    match = re.search(pattern, cif_str)
    if match:
        chemical_formula = match.group(1)
        chemical_formula = chemical_formula.replace("'", "").replace(" ", "")

        modified_cif = re.sub(pattern_2, r'\1' + chemical_formula + r'\3', cif_str)

        return modified_cif
    else:
        raise Exception(f"Chemical formula not found {cif_str}")


def add_atomic_props_block(cif_str, oxi=False):
    comp = Composition(extract_formula_nonreduced(cif_str))

    block = get_atomic_props_block(composition=comp, oxi=oxi)

    # the hypothesis is that the atomic properties should be the first thing
    #  that the model must learn to associate with the composition, since
    #  they will determine so much of what follows in the file
    pattern = r"_symmetry_space_group_name_H-M"
    match = re.search(pattern, cif_str)

    if match:
        start_pos = match.start()
        modified_cif = cif_str[:start_pos] + block + "\n" + cif_str[start_pos:]
        return modified_cif
    else:
        raise Exception(f"Pattern not found: {cif_str}")


def round_numbers(cif_str, decimal_places=4):
    # Pattern to match a floating point number in the CIF file
    # It also matches numbers in scientific notation
    pattern = r"[-+]?\d*\.\d+([eE][-+]?\d+)?"

    # Function to round the numbers
    def round_number(match):
        number_str = match.group()
        number = float(number_str)
        # Check if number of digits after decimal point is less than 'decimal_places'
        if len(number_str.split('.')[-1]) <= decimal_places:
            return number_str
        rounded = round(number, decimal_places)
        return format(rounded, '.{}f'.format(decimal_places))

    # Replace all occurrences of the pattern using a regex sub operation
    cif_string_rounded = re.sub(pattern, round_number, cif_str)

    return cif_string_rounded
