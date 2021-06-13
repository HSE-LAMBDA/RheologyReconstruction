import os
import shutil
import requests
from tqdm import tqdm

from obspy.io.segy.core import _read_segy

import numpy as np
import matplotlib.pyplot as plt


def get_marmousi2_data_location():

    # download link and friends
    SOURCE_URL = "https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz"
    PACKED_FILE_NAME = SOURCE_URL.split('/')[-1]
    FOLDER_NAME = PACKED_FILE_NAME.split('.')[0]

    # model files names (we know them from MARMOUSI2 spec)
    DENSITY_FILE = os.path.join(FOLDER_NAME, "model", "MODEL_DENSITY_1.25m.segy")
    P_WAVE_VELOCITY_FILE = os.path.join(FOLDER_NAME, "model", "MODEL_P-WAVE_VELOCITY_1.25m.segy")
    S_WAVE_VELOCITY_FILE = os.path.join(FOLDER_NAME, "model", "MODEL_S-WAVE_VELOCITY_1.25m.segy")

    return (SOURCE_URL, PACKED_FILE_NAME, FOLDER_NAME, DENSITY_FILE, P_WAVE_VELOCITY_FILE, S_WAVE_VELOCITY_FILE)


def get_marmousi2_model_params():

    # space steps, in meters (we know them from MARMOUSI2 spec)
    DX = 1.25
    DZ = 1.25

    # number of space steps
    NUM_X = 13601
    NUM_Z = 2801

    return (DX, DZ, NUM_X, NUM_Z)


def download(url: str, fname: str):
    resp = requests.get(url, stream=True)
    total = int(resp.headers.get('content-length', 0))
    with open(fname, 'wb') as file, tqdm(
            desc=fname,
            total=total,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=1024):
            size = file.write(data)
            bar.update(size)


def get_and_unpack_data():
    print("Local data not found! We are going to prepare it. Please, be patient, it will take some time.")

    (SOURCE_URL, PACKED_FILE_NAME, FOLDER_NAME, DENSITY_FILE, P_WAVE_VELOCITY_FILE, S_WAVE_VELOCITY_FILE) = get_marmousi2_data_location()

    print("Downloading elastic marmousi model")
    download(SOURCE_URL, PACKED_FILE_NAME)

    print("Unpacking elastic marmousi model")
    shutil.unpack_archive(PACKED_FILE_NAME)

    for f in [DENSITY_FILE, P_WAVE_VELOCITY_FILE, S_WAVE_VELOCITY_FILE]:
        shutil.unpack_archive(f + ".tar.gz", os.path.split(f)[0])
    print("Done")


def read_data(start_z = 0.0, stop_z = 3500.0, start_x = 0.0, stop_x = 17000.0, coarse_factor = 1):

    (SOURCE_URL, PACKED_FILE_NAME, FOLDER_NAME, DENSITY_FILE, P_WAVE_VELOCITY_FILE, S_WAVE_VELOCITY_FILE) \
        = get_marmousi2_data_location()

    if not os.path.exists(FOLDER_NAME):
        get_and_unpack_data()

    (DX, DZ, NUM_X, NUM_Z) = get_marmousi2_model_params()

    rho_coeffs = np.zeros((NUM_Z, NUM_X))
    cp_coeffs = np.zeros((NUM_Z, NUM_X))
    cs_coeffs = np.zeros((NUM_Z, NUM_X))

    start_x_ind = int(start_x / DX)
    stop_x_ind = 1 + int(stop_x / DX)
    start_z_ind = int(start_z / DZ)
    stop_z_ind = 1 + int(stop_z / DZ)

    _ = [rho_coeffs, cp_coeffs, cs_coeffs]

    for q, f in enumerate([DENSITY_FILE, P_WAVE_VELOCITY_FILE, S_WAVE_VELOCITY_FILE]):
        print("Reading", f)
        segy = _read_segy(f)
        for i, tr in enumerate(segy.traces):
            _[q][:, i] = tr.data

    # clip & coarse & inverse Z-axis
    rho_coeffs = rho_coeffs[stop_z_ind:start_z_ind:-coarse_factor, start_x_ind:stop_x_ind:coarse_factor]
    cp_coeffs = cp_coeffs[stop_z_ind:start_z_ind:-coarse_factor, start_x_ind:stop_x_ind:coarse_factor]
    cs_coeffs = cs_coeffs[stop_z_ind:start_z_ind:-coarse_factor, start_x_ind:stop_x_ind:coarse_factor]

    rho_coeffs *= 1000

    mu_coeffs = rho_coeffs * np.square(cs_coeffs)
    la_coeffs = rho_coeffs * (np.square(cp_coeffs) - 2 * np.square(cs_coeffs))

    return (rho_coeffs, cp_coeffs, cs_coeffs, la_coeffs, mu_coeffs)


def show(data, title, start_z = 0.0, stop_z = 3500.0, start_x = 0.0, stop_x = 17000.0):
    nz, nx = data.shape

    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    im = ax.pcolormesh(np.linspace(start_x, stop_x, nx), np.linspace(-stop_z, -start_z, nz), data)
    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_title(title)

    fig.tight_layout()
    plt.show()
