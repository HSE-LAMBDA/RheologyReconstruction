import os
import numpy as np
import torch
import torch.nn as nn


def check_if_dir_exists(path):
	"""
	A bit more robust than os.path.isdir
	"""

	if not os.path.exists(path): return False
	
	if os.path.isfile(path):
		print(f'{path} exists as a file') 
		return False

	if os.path.isdir(path):
		if len(os.listdir(path)) > 0:
			print(f'Found {path} as a non-empty directory with {len(os.listdir(path))}'
				  f' files')
		return True


def make_snapshot_directory(path):

	if path is None: path = 'snapshots'

	if not os.path.isabs(path): path = os.path.join(os.getcwd(), path)

	if not check_if_dir_exists(path):
		if os.path.isfile(path): raise OSError(f'{path} already exists as a file')
		os.mkdir(path)
	
	return path


def get_latest_snapshot_name(path):
    if not os.path.isabs(path): path = os.path.join(os.getcwd(), path)
    snapshots = [os.path.join(path, s) for s in os.listdir(path)]

    if not snapshots: raise RuntimeError('No snapshots found')
    latest_snapshot = max(snapshots, key=os.path.getctime)
    
    return latest_snapshot


def number_of_parameters(model: nn.Module) -> int:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of parameters: {}".format(total))
    print("Trainable number of parameters: {}".format(trainable))


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


__running_on_notebook__ = isnotebook()

if __running_on_notebook__: 
    from tqdm import tqdm as tqdm
else:
    from tqdm.notebook import tqdm as tqdm
