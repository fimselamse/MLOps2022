import os

import numpy as np
import torch
from torch.nn.functional import normalize
from torch.utils.data import TensorDataset


def main():
    dir = os.path.dirname(__file__)

    # training data is separated into multiple files - load and concatenate.
    images, labels = [], []
    for i in range(5):
        traindata = np.load(
            os.path.abspath(
                os.path.join(dir, f"../../data/raw/corruptmnist/train_{i}.npz")
            )
        )
        images.append(traindata["images"])
        labels.append(traindata["labels"])

    images = torch.FloatTensor(np.concatenate(images))
    labels = torch.LongTensor(np.concatenate(labels))
    normalize(images)
    trainset = TensorDataset(images, labels)
    torch.save(
        trainset,
        os.path.abspath(os.path.join(dir, "../../data/processed/train_mnist.pt")),
    )

    # load and transform test
    testdata = np.load(
        os.path.abspath(os.path.join(dir, "../../data/raw/corruptmnist/test.npz"))
    )
    images = testdata["images"]
    labels = testdata["labels"]
    test_images = torch.FloatTensor(testdata["images"])
    test_labels = torch.LongTensor(testdata["labels"])
    normalize(test_images)
    testset = TensorDataset(test_images, test_labels)
    torch.save(
        testset,
        os.path.abspath(os.path.join(dir, "../../data/processed/test_mnist.pt")),
    )


if __name__ == "__main__":
    main()
