# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
import os
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.functional import normalize


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
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
        os.path.abspath(os.path.join(dir, f"../../data/processed/train_mnist.pt")),
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
    testset = TensorDataset(images, labels)
    torch.save(
        testset,
        os.path.abspath(os.path.join(dir, f"../../data/processed/test_mnist.pt")),
    )


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
