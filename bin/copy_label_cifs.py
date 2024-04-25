import os
import tarfile
import shutil
import re
import tqdm

def copy_matching_cif_files(cif_directory, tar_gz_path, destination_path):
    
    # Open the tar.gz file and collect the names of the members
    system_ids = set()
    with tarfile.open(tar_gz_path, "r:gz") as tar:
        for member in tar.getmembers():
            file_name = member.name.split('/')[-1]
            if file_name.endswith('.cif'):
                #if "miller" in file_name:
                sid = re.match(r'random\d+', file_name).group(0)
                system_ids.add(sid)
    # breakpoint()

    # # Ensure destination directory exists, if not, create it
    # if not os.path.exists(destination_directory):
    #     os.makedirs(destination_directory)

    # print(f"Copying CIF files from {cif_directory} to {destination_directory}...")
    # for sid in tqdm.tqdm(system_ids):
    #     source_path = os.path.join(cif_directory, f"{sid}.cif")
    #     destination_path = os.path.join(destination_directory, f"{sid}.cif")
    #     shutil.copy(source_path, destination_path)
    #     print(f"Copied: {sid} to {destination_directory}")

    # Prepare to create the destination tar.gz file
    with tarfile.open(destination_path, "w:gz") as tar_dest:
        # For each system ID, try to find and add the corresponding CIF file
        for sid in system_ids:
            source_path = os.path.join(cif_directory, f"{sid}.cif")
            if os.path.exists(source_path):
                tar_dest.add(source_path, arcname=os.path.basename(source_path))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Copy CIF files from a directory to another directory based on the system IDs in a tar.gz file.")
    parser.add_argument("cif_dir", type=str, help="The directory containing CIF files.")
    parser.add_argument("tar_gz_path", type=str, help="The path to the tar.gz file.")
    parser.add_argument("dest_path", type=str, help="The destination path.")
    args = parser.parse_args()

    copy_matching_cif_files(args.cif_dir, args.tar_gz_path, args.dest_path)

