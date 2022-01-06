import logging
import os

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from model import CNN, Linear
from torch import nn, optim

log = logging.getLogger(__name__)


@hydra.main(config_path="../conf", config_name="config")
def train(cfg):
    log.info(f"Training started with parameters: {cfg.params}")
    torch.manual_seed(cfg.params.seed)

    # TODO: Implement training loop here
    if cfg.params.model == "cnn":
        model = CNN()
    else:
        model = Linear()

    model.train()
    train_set = torch.load(f"{cfg.paths.data_path}/{cfg.files.train_data}")
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=cfg.params.batch_size, shuffle=True
    )
    log.info("Training data loaded")

    test_set = torch.load(f"{cfg.paths.data_path}/{cfg.files.test_data}")

    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=cfg.params.batch_size, shuffle=False
    )

    log.info("Test data loaded")

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=cfg.params.lr)
    step = 0
    train_losses, test_losses, accuracies = [], [], []
    for e in range(cfg.params.epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()

            out = model(images)

            loss = criterion(out, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            step += 1
            # checkpointing
            # if step % 100 == 0:
            #     os.makedirs("models/", exist_ok=True)
            #     torch.save(model.state_dict(), "models/trained_model.pt")

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
            epoch_loss = running_loss / len(trainloader)
            epoch_val_loss = running_val_loss / len(testloader)
            epoch_val_acc = running_accuracy / len(testloader)

            train_losses.append(epoch_loss)
            test_losses.append(epoch_val_loss)
            accuracies.append(epoch_val_acc)

            logging.info(f"Testset accuracy: {epoch_val_acc*100}%")
            logging.info(f"Validation loss: {epoch_val_loss}")
            logging.info(f"Training loss: {epoch_loss}")
    logging.info("Training finished!")

    # saving final model
    os.makedirs("models/", exist_ok=True)
    torch.save(model.state_dict(), "models/trained_model.pt")
    logging.info("Model saved")

    plt.plot(np.arange(cfg.params.epochs), train_losses, label="training loss")
    plt.plot(np.arange(cfg.params.epochs), test_losses, label="validation loss")
    plt.plot(np.arange(cfg.params.epochs), accuracies, label="accuracy")
    plt.xlabel("epochs")
    plt.legend()
    plt.title("MNIST model training")

    os.makedirs("reports/figures/", exist_ok=True)
    plt.savefig("reports/figures/train_loss.png")


if __name__ == "__main__":
    with torch.profiler.profile() as profiler:
        train()
