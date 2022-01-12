import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import CNN, Linear
from torch import nn, optim
from pytorch_lightning import LightningModule, Trainer, seed_everything
import torch.nn.functional as F


class Linear(LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 10)
        
        self.lr = lr

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)

        return x
    
    def training_step(self, batch, batch_idx):
        image, label = batch
        out = self(image)
        loss = F.nll_loss(out, label)
        return loss
        
    def validation_step(self, batch, batch_idx):
        image, label = batch
        out = self(image)
        val_loss = F.nll_loss(out, label)
        return val_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


@hydra.main(config_path="../conf", config_name="config")
def main(cfg):    
    seed_everything(cfg.params.seed, workers=True)
    
    model = Linear(lr=cfg.params.lr)
    
    train_set = torch.load(f"{cfg.paths.data_path}/{cfg.files.train_data}")
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.params.batch_size, shuffle=True, num_workers=4
    )

    test_set = torch.load(f"{cfg.paths.data_path}/{cfg.files.test_data}")
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.params.batch_size, shuffle=False, num_workers=4
    )
    
    trainer = Trainer(
        accelerator='auto',
        max_epochs=cfg.params.epochs,
    )
    
    trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=testloader)


if __name__ == "__main__":
    
    main()
    
    
    