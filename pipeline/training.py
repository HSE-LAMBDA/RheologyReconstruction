import torch
import random
import numpy as np
import os

from dataset import SeismogramDataset
from neural_networks.segnet import SegNet_3Head
from trainer import BaseTrainer

SEED = 0


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = 'cpu'
    ROOT_DIR = os.path.join(os.getcwd())
    dataset_path = os.path.join(ROOT_DIR, 'dolfin_adjoint', '2_subdomains')
    train_dataset = SeismogramDataset(dataset_path)
    model = SegNet_3Head()
    detector_coords = [(np.array([c, 2000.])) for c in np.linspace(0., 2000., 128)]  # TODO: research

    # TODO: test with nonempty logger
    t = BaseTrainer(
        model,
        device,
        train_dataset,
        optimizer_type=torch.optim.Adam,
        optimizer_params={'lr': 1e-3},
        snapshot_interval=250
    )

    # TODO: test if everything is allright with batch_size > 1
    t.train(detector_coords, batch_size=1, epochs=100, num_solver_type='dolfin_adjoint')


if __name__ == '__main__':
    main()
