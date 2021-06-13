from marmousi.marmousi2_tools import read_data, show

if __name__ == '__main__':
    start_z = 462.5
    stop_z = 3500.0
    start_x = 2000
    stop_x = 8000
    coarse_factor = 10

    rho_coeffs, cp_coeffs, cs_coeffs, la_coeffs, mu_coeffs = read_data(start_z = start_z, stop_z = stop_z,
                                                                       start_x = start_x, stop_x = stop_x,
                                                                       coarse_factor = coarse_factor)
    print(rho_coeffs.shape)

    show(rho_coeffs, 'Density, kg/m3', start_z = start_z, stop_z = stop_z, start_x = start_x, stop_x = stop_x)
    show(cp_coeffs, 'Cp, m/s', start_z = start_z, stop_z = stop_z, start_x = start_x, stop_x = stop_x)
    show(cs_coeffs, 'Cs, m/s', start_z = start_z, stop_z = stop_z, start_x = start_x, stop_x = stop_x)
    show(la_coeffs, 'la, Pa', start_z = start_z, stop_z = stop_z, start_x = start_x, stop_x = stop_x)
    show(mu_coeffs, 'mu, Pa', start_z = start_z, stop_z = stop_z, start_x = start_x, stop_x = stop_x)