import os
import glob
import argparse
import tqdm
from pymatgen.io.cif import CifParser
from ase import Atoms
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms

def read_cif_file(file_path):
    """Read CIF file using pymatgen and convert to ASE atoms object."""
    sid = os.path.splitext(os.path.basename(file_path))[0]
    #breakpoint()
 
    structure = CifParser(file_path).get_structures()[0]
    ase_atoms = Atoms(symbols=[site.specie.symbol for site in structure],
                      positions=[site.coords for site in structure],
                      cell=structure.lattice.matrix,
                      pbc=True)
    return sid, ase_atoms
    # except:
    #     print(f"Error reading CIF file: {file_path}")
    #     return sid, None
    

def visualize_atomic_structure(sid, ase_atoms, save_path, orientation=("-75x, 45y, 10z")):
    """Visualize atomic structure and optionally save to a file."""
    fig, ax = plt.subplots()
    plot_atoms(ase_atoms, ax=ax, rotation=orientation)
    file_name = os.path.join(save_path, f"{sid}.png")
    plt.savefig(file_name, dpi=300)
    plt.close(fig)

def process_files(file_paths, save_path):
    """Process list of CIF files."""
    if not file_paths:
        raise FileNotFoundError("No CIF files found.")
    for file_path in tqdm.tqdm(file_paths):
        try:
            sid, ase_atoms = read_cif_file(file_path)
            visualize_atomic_structure(sid, ase_atoms, save_path)
        except:
            print(f"Error processing CIF file: {file_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--file_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    if os.path.isdir(args.file_path):
        file_paths = glob.glob(os.path.join(args.file_path, "*.cif"))
    elif os.path.isfile(args.file_path):
        file_paths = [args.file_path]
    else:
        raise FileNotFoundError(f"No valid file or directory at {args.file_path}")

    process_files(file_paths, args.save_path)
