from lib import get_unit_cell_volume


if __name__ == '__main__':

    a = 7.03059449
    b = 13.81160316
    c = 7.11463800
    alpha_deg = 90
    beta_deg = 106.45293817
    gamma_deg = 90

    volume = get_unit_cell_volume(a, b, c, alpha_deg, beta_deg, gamma_deg)

    print("Unit cell volume:", volume)
