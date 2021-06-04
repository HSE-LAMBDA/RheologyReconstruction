import os
import shutil
import requests
from tqdm import tqdm

from obspy.io.segy.core import _read_segy

import numpy as np

import matplotlib.pyplot as plt


# download link and friends
SOURCE_URL = "https://s3.amazonaws.com/open.source.geoscience/open_data/elastic-marmousi/elastic-marmousi-model.tar.gz"
PACKED_FILE_NAME = SOURCE_URL.split('/')[-1]
FOLDER_NAME = PACKED_FILE_NAME.split('.')[0]

# model files names (we know them from MARMOUSI2 spec)
DENSITY_FILE = os.path.join(FOLDER_NAME, "model", "MODEL_DENSITY_1.25m.segy")
P_WAVE_VELOCITY_FILE = os.path.join(FOLDER_NAME, "model", "MODEL_P-WAVE_VELOCITY_1.25m.segy")
S_WAVE_VELOCITY_FILE = os.path.join(FOLDER_NAME, "model", "MODEL_S-WAVE_VELOCITY_1.25m.segy")

# space steps, in meters (we know them from MARMOUSI2 spec)
DX = 1.25
DZ = 1.25

# number of space steps
NUM_X = 13601
NUM_Z = 2801


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

    print("Downloading elastic marmousi model")
    download(SOURCE_URL, PACKED_FILE_NAME)

    print("Unpacking elastic marmousi model")
    shutil.unpack_archive(PACKED_FILE_NAME)

    for f in [DENSITY_FILE, P_WAVE_VELOCITY_FILE, S_WAVE_VELOCITY_FILE]:
        shutil.unpack_archive(f + ".tar.gz", os.path.split(f)[0])
    print("Done")


def read_data():
    if not os.path.exists(FOLDER_NAME):
        get_and_unpack_data()

    rho_coeffs = np.zeros((NUM_Z, NUM_X))
    cp_coeffs = np.zeros((NUM_Z, NUM_X))
    cs_coeffs = np.zeros((NUM_Z, NUM_X))

    _ = [rho_coeffs, cp_coeffs, cs_coeffs]

    for q, f in enumerate([DENSITY_FILE, P_WAVE_VELOCITY_FILE, S_WAVE_VELOCITY_FILE]):
        print("Reading", f)
        segy = _read_segy(f)
        for i, tr in enumerate(segy.traces):
            _[q][::-1, i] = tr.data

    rho_coeffs *= 1000
    return (rho_coeffs, cp_coeffs, cs_coeffs)


def show(data, title):
    fig, ax = plt.subplots()
    ax.set_aspect('equal')

    im = ax.pcolormesh(np.linspace(0.0, (NUM_X - 1) * DX, NUM_X), np.linspace(-(NUM_Z - 1) * DZ, 0.0, NUM_Z), data)
    fig.colorbar(im, ax=ax, orientation='horizontal')
    ax.set_title(title)

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    rho_coeffs, cp_coeffs, cs_coeffs = read_data()
    show(rho_coeffs, 'Density, kg/m3')
    show(cp_coeffs, 'Cp, m/s')
    show(cs_coeffs, 'Cs, m/s')