import torch
import numpy as np
from tests import _PATH_DATA

train_set = torch.load(f'{_PATH_DATA}/processed/train_mnist.pt')
test_set = torch.load(f'{_PATH_DATA}/processed/test_mnist.pt')

# processed data is stored as torch.utils.data.TensorDataset and thus
# needs to be accessed a little different. It works though.

def test_data_size_and_shape():
    assert list(train_set[:][0].shape) == [25000, 28, 28]
    assert list(test_set[:][0].shape) == [5000, 28, 28]
    
def test_labels():
    assert np.all(np.arange(10) == np.unique(train_set[:][1]))
    assert np.all(np.arange(10) == np.unique(test_set[:][1]))