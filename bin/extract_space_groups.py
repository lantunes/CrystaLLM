from tqdm import tqdm

try:
    import cPickle as pickle
except ImportError:
    import pickle


if __name__ == '__main__':
    fname = "../out/matproj_all_2022_04_12.cif.pkl"

    with open(fname, "rb") as f:
        data = pickle.load(f)

    cif_lines = []

    for cif in tqdm(data):
        lines = cif.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith("_symmetry_space_group_name_H-M"):
                cif_lines.append(line)

    space_groups = set()

    for cif_line in tqdm(cif_lines):
        space_groups.add(cif_line.split("_symmetry_space_group_name_H-M")[1].strip())

    for sg in space_groups:
        print(sg)

    print(len(space_groups))
