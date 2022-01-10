import torch
import numpy as np
from model import CNN, Linear



def test_cnn():
    model = CNN()
    
    input = torch.rand(1, 28, 28)
    output = model(input)
    
    assert list(output.shape) == [1, 10]
    
def test_linear():
    model = Linear()
    
    input = torch.rand(1, 28, 28)
    output = model(input)
    
    assert list(output.shape) == [1, 10]