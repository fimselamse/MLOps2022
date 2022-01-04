import argparse
import sys, os

sys.path.append(os.path.abspath(".."))

import torch
from torch import nn, optim
from model import MyAwesomeModel
import matplotlib.pyplot as plt
import numpy as np


class PredictModel(object):
    """Helper class that will help launch class methods as commands
    from a single script
    """

    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python train_model.py <command>",
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print("Unrecognized command")

            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()

    def predict(self):
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--model", default="", type=str)
        parser.add_argument("--data", default="", type=str)
        args = parser.parse_args(sys.argv[2:])
        print(args)

        dir = os.path.dirname(__file__)

        model = MyAwesomeModel()
        state_dict = torch.load(
            os.path.abspath(os.path.join(dir, "../../" + args.model))
        )
        model.load_state_dict(state_dict)
        model.eval()

        images = torch.torch.load(
            os.path.abspath(os.path.join(dir, "../../" + args.data))
        )
        loader = torch.utils.data.DataLoader(images, batch_size=1, shuffle=True)
        predictions = []

        for item, label in loader:
            with torch.no_grad():
                prediction = model(item)

                predicted_class = np.argmax(prediction)

                predictions.append([item, predicted_class])

        return predictions


if __name__ == "__main__":
    PredictModel()
