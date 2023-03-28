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
            if len(line) > 0 and not line.startswith("#"):
                cif_lines.append(line)

    keywords = set()

    for cif_line in tqdm(cif_lines):
        tokens = cif_line.split(" ")
        for t in tokens:
            if t.startswith("_"):
                keywords.add(t)

    for kw in keywords:
        print(kw)

    print(len(keywords))
