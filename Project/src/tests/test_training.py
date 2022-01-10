import torch
import numpy as np
from model import CNN, Linear
from train_model import train

def test_training():
    model = CNN()
    
    # training network not really set up for this kind of test.
    # since it uses hydra and a config file, most errors will be logged 
    # at runtime
    assert 2==1
    