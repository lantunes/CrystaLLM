
class CIFData:
    def __init__(self):
        self.composition = None
        self.space_group = None
        self.space_group_num = None
        self.cell_length_a = None
        self.cell_length_b = None
        self.cell_length_c = None
        self.cell_angle_alpha = None
        self.cell_angle_beta = None
        self.cell_angle_gamma = None
        self.cell_volume = None

    def __repr__(self):
        return f"CIFData[" \
               f"comp:'{self.composition}', " \
               f"spacegroup:'{self.space_group}', " \
               f"spacegroup num:'{self.space_group_num}', " \
               f"a:{self.cell_length_a}, " \
               f"b:{self.cell_length_b}, " \
               f"c:{self.cell_length_c}, " \
               f"alpha:{self.cell_angle_alpha}, " \
               f"beta:{self.cell_angle_beta}, " \
               f"gamma:{self.cell_angle_gamma}, " \
               f"vol:{self.cell_volume}, " \
               f"]"

    def to_csv_row(self):
        return [
            self.composition,
            self.space_group,
            self.space_group_num,
            self.cell_length_a,
            self.cell_length_b,
            self.cell_length_c,
            self.cell_angle_alpha,
            self.cell_angle_beta,
            self.cell_angle_gamma,
            self.cell_volume
        ]

    @staticmethod
    def from_csv_row(row):
        data = CIFData()
        data.composition = row[0]
        data.space_group = row[1]
        data.space_group_num = int(row[2])
        data.cell_length_a = float(row[3])
        data.cell_length_b = float(row[4])
        data.cell_length_c = float(row[5])
        data.cell_angle_alpha = float(row[6])
        data.cell_angle_beta = float(row[7])
        data.cell_angle_gamma = float(row[8])
        data.cell_volume = float(row[9])
        return data

    @staticmethod
    def csv_columns():
        return [
            "composition",
            "space_group",
            "space_group_num",
            "cell_length_a",
            "cell_length_b",
            "cell_length_c",
            "cell_angle_alpha",
            "cell_angle_beta",
            "cell_angle_gamma",
            "cell_volume"
        ]

    def is_valid(self):
        return self.composition is not None and \
               self.space_group is not None and \
               self.space_group_num is not None and \
               self.cell_length_a is not None and \
               self.cell_length_b is not None and \
               self.cell_length_c is not None and \
               self.cell_angle_alpha is not None and \
               self.cell_angle_beta is not None and \
               self.cell_angle_gamma is not None and \
               self.cell_volume is not None


def populate_cif_data(cif_data, cif, validate=False):
    for line in cif.split("\n"):
        if line.startswith("data_"):
            cif_data.composition = line.split("_")[1]
        elif line.startswith("_symmetry_space_group_name_H-M"):
            cif_data.space_group = line.split(" ", 1)[1].strip()
        elif line.startswith("_symmetry_Int_Tables_number"):
            cif_data.space_group_num = int(line.split(" ")[1].strip())
        elif line.startswith("_cell_length_a"):
            cif_data.cell_length_a = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_length_b"):
            cif_data.cell_length_b = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_length_c"):
            cif_data.cell_length_c = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_angle_alpha"):
            cif_data.cell_angle_alpha = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_angle_beta"):
            cif_data.cell_angle_beta = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_angle_gamma"):
            cif_data.cell_angle_gamma = float(line.split(" ")[1].strip())
        elif line.startswith("_cell_volume"):
            cif_data.cell_volume = float(line.split(" ")[1].strip())

    if validate and not cif_data.is_valid():
        raise Exception("invalid CIFData")
