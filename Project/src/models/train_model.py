import argparse
import sys, os

sys.path.append(os.path.abspath(".."))

import torch
from torch import nn, optim
from model import CNN
import matplotlib.pyplot as plt
import numpy as np


class TrainModel(object):
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

    def train(self):
        parser = argparse.ArgumentParser(description="Training arguments")
        parser.add_argument("--lr", default=0.001, type=float)
        parser.add_argument("--epochs", default=100, type=int)
        parser.add_argument("--batchsize", default=128, type=int)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)

        dir = os.path.dirname(__file__)

        # TODO: Implement training loop here
        model = CNN()
        model.train()
        train_set = torch.load(
            os.path.abspath(os.path.join(dir, "../../data/processed/train_mnist.pt"))
        )
        trainloader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batchsize, shuffle=True
        )

        test_set = torch.load(
            os.path.abspath(os.path.join(dir, "../../data/processed/test_mnist.pt"))
        )
        testloader = torch.utils.data.DataLoader(
            test_set, batch_size=args.batchsize, shuffle=True
        )

        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        step = 0
        train_losses, test_losses = [], []
        for e in range(args.epochs):
            running_loss = 0
            for images, labels in trainloader:
                optimizer.zero_grad()

                out = model(images)

                loss = criterion(out, labels)
                loss.backward()

                optimizer.step()

                running_loss += loss.item()
                train_losses.append(loss.item())
                step += 1
                if step % 100 == 0:
                    torch.save(
                        model.state_dict(),
                        os.path.abspath(
                            os.path.join(dir, "../../models/checkpoint.pth")
                        ),
                    )
            else:
                with torch.no_grad():
                    running_accuracy = 0
                    running_val_loss = 0
                    for images, labels in testloader:
                        out = model(images)
                        loss = criterion(out, labels)
                        running_val_loss += loss.item()
                        top_p, top_class = out.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy = torch.mean(equals.type(torch.FloatTensor))
                        running_accuracy += accuracy.item()
                print(f"Testset accuracy: {running_accuracy/len(testloader)*100}%")
                print(f"Validation loss: {running_val_loss/len(testloader)}")
                print(f"Training loss: {running_loss/len(trainloader)}")

        plt.plot(np.arange(step), train_loss)
        plt.savefig(
            os.path.abspath(os.path.join(dir, "../reports/figures/train_loss.png"))
        )


if __name__ == "__main__":
    TrainModel()
