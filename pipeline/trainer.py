#package 

import torch
import torch.nn as nn
import numpy as np
import os
import time 
import torch.nn.functional as F
import torch.optim as optim

from datetime import datetime
from dataset import SeismogramBatch
from tqdm.notebook import tqdm
from utils import get_latest_snapshot_name, make_snapshot_directory
from neural_networks.loss import weightedBCELoss
from IPython.display import clear_output



class BaseTrainer:

    def __init__(
        self,
        model,
        device, 
        train_dataset,
        val_dataset=None,
        loss_fn=weightedBCELoss,
        optimizer_type=optim.Adam,
        optimizer_params=None,
        gradient_clipping=None,
        logger=None,
        snapshot_path=None,
        snapshot_interval=1000,     
    ):

        """ 
        :param model: a model to train
        :type model : torch.nn.Module

        :param loss_fn: a differentiable loss function for model training
        :returns loss function: torch.tensor

        :type train_dataset: SeismogramDataset
        :type val_dataset  : SeismogramDataset

        """

        self.model   = model
        self.device  = device
        self.loss_fn = loss_fn()
        
        self.train_dataset  = train_dataset
        self.val_dataset    = val_dataset
        self.optimizer_type = optimizer_type

        if optimizer_params is None:
            self.optimizer  = self.optimizer_type(self.model.parameters())
        else:
            self.optimizer  = self.optimizer_type(self.model.parameters(), **optimizer_params)

        self.gradient_clipping = gradient_clipping 

        self.logger        = logger
        #if self.logger is not None: self.logger.setup_loss_fn(self.loss_fn)
        
        self.snapshot_path     = make_snapshot_directory(snapshot_path)
        self.snapshot_interval = snapshot_interval

        # internal snapshot parameters
        self.date_format       = '%Y-%m-%d_%H-%M-%S'


    def load_latest_snapshot(self):

        sname    = get_latest_snapshot_name(self.snapshot_path)
        snapshot = torch.load(sname)

        error_msg_header = f'Error loading snapshot {sname}' +\
                            '- incompatible snapshot format. '
        if 'optimizer' not in snapshot:
            raise KeyError(error_msg_header + 'Key "optimizer" is missing')
        if 'model' not in snapshot:
            raise KeyError(error_msg_header + 'Key "model" is missing')

        self.model.load_state_dict(snapshot['model'])
        self.optimizer.load_state_dict(snapshot['optimizer'])


    def save_model(self, replace_latest=False):

        if self.snapshot_path is None: return
                    
        time_string = datetime.now().strftime(self.date_format)

        states = {
            'model'  : self.model.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

        if not replace_latest:
            torch.save(states, os.path.join(self.snapshot_path, time_string + '.pth'))
        else:
            try:
                os.remove(get_latest_snapshot_name(self.snapshot_path))
            except Exception:
                pass
            torch.save(states, os.path.join(self.snapshot_path, time_string + '.pth'))


    def validate(self, metrics_list):
        """
        Evaluate the list of given metrics with data.
        If validational dataset not is specified, evaluation will run on the single batch from
        training dataset
 
        :type metrics_list: List of :class:`BaseMetric` or None 

        """

        self.model.train(False)

        batch_size = 64

        if self.val_dataset is None:

            batch_gen = torch.utils.data.DataLoader(
                self.train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                num_workers=5, collate_fn=lambda x: SeismogramBatch(x)
            )

        else:
            batch_gen = torch.utils.data.DataLoader(
                self.val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True,
                num_workers=5, collate_fn=lambda x: SeismogramBatch(x)
            )


        m_values = {m.name: [] for m in metrics_list}

        with torch.no_grad():

            if self.val_dataset is None:
            
                val_batch = next(iter(batch_gen)).to(self.device)
                preds = self.model.forward(val_batch)

                for m in metrics_list:
                    m_values[m.name].append(m(
                        preds.cpu().data.numpy(),
                        val_batch.masks.cpu().data.numpy(),
                        val_batch.weights.cpu().data.numpy()
                    ))

            else:
            	
                for val_batch in batch_gen:

                    val_batch = val_batch.to(self.device)
                    preds = self.model.forward(val_batch)

                    for m in metrics_list:
                        m_values[m.name].append(m(
                            preds.cpu().data.numpy(),
                            val_batch.masks.cpu().data.numpy(),
                            val_batch.weights.cpu().data.numpy()
                        ))

        for k in m_values.keys(): m_values[k] = np.mean(np.array(m_values[k]))

        return m_values


    def train(self, batch_size=32, epochs=100, from_zero=True):      

        self.model = self.model.to(self.device)
        if not from_zero: self.load_latest_snapshot()

        train_batch_gen = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, 
            shuffle=True, pin_memory=True, num_workers=5,
            collate_fn=lambda x: SeismogramBatch(x)
        )


       	step = 1


        for current_epoch in tqdm(range(epochs), desc=f'Running training procedure'):

            self.model.train(True)

            for batch in tqdm(train_batch_gen, desc=f'Epoch {current_epoch + 1} of {epochs}'): 

                self.model.zero_grad()
                self.optimizer.zero_grad()

                batch  = batch.to(self.device)
                preds  = self.model.forward(batch.seismograms)

                """
                Предсказания сети необходимо 
                1) Перенести на неравноменую сетку
                2) Загрузить эту сетку в fenics
                3) С его помощью посчитать градиент от функционала ошибки по выходу сети
                4) Применить chain rule
        		"""

                #TODO: implement
                L, dL_dm = get_gradient_from_solver(preds.cpu().data.numpy())

                preds.backward(dL_dm)

                self.optimizer.step()
                torch.cuda.empty_cache()

                step += 1

                self.logger.log(self, step, L.item())
                if step % self.snapshot_interval == 0 : self.save_model()

            clear_output(wait=True)
