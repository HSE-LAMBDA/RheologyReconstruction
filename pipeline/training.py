import torch
import random
import numpy as np
import os

from dataset import SeismogramDataset
from neural_networks.segnet import SegNet_3Head

from utils import number_of_parameters
from trainer import BaseTrainer


def main():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    ROOT_DIR = os.path.join(os.getcwd())

    dataset_path = os.path.join(ROOT_DIR, 'dolfin_adjoint', '2_subdomains')
    train_dataset = SeismogramDataset(dataset_path)

    model = SegNet_3Head()
    number_of_parameters(model)

    detector_coords = [(np.array([c, 2000.])) for c in np.linspace(0., 2000., 128)]

    t = BaseTrainer(model,
                    device,
                    train_dataset,
                    optimizer_type=torch.optim.Adam,
                    optimizer_params={'lr': 1e-3},
                    snapshot_interval=10)

    t.train(detector_coords, batch_size=1, epochs=100, num_solver_type='dolfin_adjoint')


if __name__ == '__main__':
    main()
