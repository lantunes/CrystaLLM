import tarfile
import ase.io
import argparse
import os, glob, re

def extract_tar_gz(file_path, extract_path):
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Ensure the extraction path exists or create it
    os.makedirs(extract_path, exist_ok=True)

    # Try to open and extract the tar.gz file
    try:
        with tarfile.open(file_path, 'r:gz') as tar:
            # Optionally list all members of the tar archive
            tar.list()

            # Extract all files from the archive into the specified directory
            tar.extractall(path=extract_path)
            print(f"Files have been successfully extracted to {extract_path}")
    except tarfile.TarError as e:
        print(f"Error extracting the tar.gz file: {e}")

def save_traj_as_cif(file_path, traj_path):
    # Load the trajectory file
    cif_files = glob.glob(os.path.join(file_path, "*.cif"))
    for file in cif_files:
        sid  = os.path.basename(file)
        unq_sid = re.match(r'random\d+', sid).group()
        traj = ase.io.read(os.path.join(traj_path, f"{unq_sid}.traj"), index=":")
        relaxed_atoms = traj[-1]
        # save ase atoms object as CIF file
        ase.io.write(os.path.join(file_path, f"{unq_sid}.cif"), relaxed_atoms)
    

def main():
    parser = argparse.ArgumentParser(description="Extract a tar.gz file to a specified location")
    parser.add_argument("file_path", type=str, help="The path to the tar.gz file to be extracted")
    parser.add_argument("extract_path", type=str, help="The directory where the files will be extracted")
    # parser.add_argument("traj_path", type=str, help="The directory where the trajectory files are stored")
    args = parser.parse_args()

    extract_tar_gz(args.file_path, args.extract_path)
    # save_traj_as_cif(args.extract_path, args.traj_path)

if __name__ == "__main__":
    main()
