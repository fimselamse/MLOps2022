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
    traindata = np.load(
        os.path.abspath(os.path.join(dir, "../../data/raw/corruptmnist/train_0.npz"))
    )
    images = traindata["images"]
    labels = traindata["labels"]

    for i in range(1, 4):
        traindata = np.load(
            os.path.abspath(
                os.path.join(dir, f"../../data/raw/corruptmnist/train_{i}.npz")
            )
        )
        images = np.concatenate((images, traindata["images"]))
        labels = np.concatenate((labels, traindata["labels"]))

    trainset = TensorDataset(torch.Tensor(images), torch.LongTensor(labels))
    torch.save(
        trainset,
        os.path.abspath(os.path.join(dir, f"../../data/processed/train_mnist.pt")),
    )

    # load and transform test
    traindata = np.load(
        os.path.abspath(os.path.join(dir, "../../data/raw/corruptmnist/test.npz"))
    )
    images = traindata["images"]
    labels = traindata["labels"]
    testset = TensorDataset(torch.Tensor(images), torch.LongTensor(labels))
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
