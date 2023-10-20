from pymatgen.analysis.chemenv.coordination_environments.chemenv_strategies import SimplestChemenvStrategy, \
    AbstractChemenvStrategy
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometries import AllCoordinationGeometries
from pymatgen.analysis.chemenv.coordination_environments.coordination_geometry_finder import LocalGeometryFinder
from pymatgen.analysis.chemenv.utils.chemenv_errors import NeighborsNotComputedChemenvError
from pymatgen.io.cif import CifParser


if __name__ == '__main__':
    cif_path = "/Users/luis/CrystalGPT-dev/out/ChallengeSet-v1/CsCuTePt/CsCuTePt_pymatgen.cif"
    distance_cutoff = 1.  # pymatgen default is 1.4
    angle_cutoff = 0.3

    lgf = LocalGeometryFinder()
    lgf.setup_parameters()

    allcg = AllCoordinationGeometries()

    strategy = SimplestChemenvStrategy(
        structure_environments=None,
        distance_cutoff=distance_cutoff,
        angle_cutoff=angle_cutoff,
        additional_condition=AbstractChemenvStrategy.AC.ONLY_ACB,
        continuous_symmetry_measure_cutoff=10,
        symmetry_measure_type=AbstractChemenvStrategy.DEFAULT_SYMMETRY_MEASURE_TYPE,
    )
    max_dist_factor = 1.5

    parser = CifParser(cif_path)
    structure = parser.get_structures()[0]

    lgf.setup_structure(structure)
    se = lgf.compute_structure_environments(maximum_distance_factor=max_dist_factor)
    strategy.set_structure_environments(se)

    for eqslist in se.equivalent_sites:
        site = eqslist[0]
        isite = se.structure.index(site)
        try:
            if strategy.uniquely_determines_coordination_environments:
                ces = strategy.get_site_coordination_environments(site)
            else:
                ces = strategy.get_site_coordination_environments_fractions(site)
        except NeighborsNotComputedChemenvError:
            continue
        if ces is None:
            continue
        if len(ces) == 0:
            continue
        comp = site.species

        if strategy.uniquely_determines_coordination_environments:
            ce = ces[0]
            if ce is None:
                continue
            thecg = allcg.get_geometry_from_mp_symbol(ce[0])
            mystring = (
                f"Environment for site #{isite} {comp.get_reduced_formula_and_factor()[0]}"
                f" ({comp}) : {thecg.name} ({ce[0]})\n"
            )
        else:
            mystring = (
                f"Environments for site #{isite} {comp.get_reduced_formula_and_factor()[0]} ({comp}) : \n"
            )
            for ce in ces:
                cg = allcg.get_geometry_from_mp_symbol(ce[0])
                csm = ce[1]["other_symmetry_measures"]["csm_wcs_ctwcc"]
                mystring += f" - {cg.name} ({cg.mp_symbol}): {ce[2]:.2%} (csm : {csm:2f})\n"

        print(mystring)
