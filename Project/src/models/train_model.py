import argparse
import sys, os

sys.path.append(os.path.abspath(".."))

import torch
from torch import nn, optim
from model import Linear, CNN
import matplotlib.pyplot as plt
import numpy as np


def train():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--lr", default=0.001, type=float)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batchsize", default=128, type=int)
    parser.add_argument("--model", default='linear', type=str)
    # add any additional argument that you want
    args = parser.parse_args(sys.argv[1:])
    print(args)

    dir = os.path.dirname(__file__)

    # TODO: Implement training loop here
    if args.model == 'CNN':
        model = CNN()
    else:
        model = Linear()
        
    model.train()
    train_set = torch.load("data/processed/train_mnist.pt")
    trainloader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batchsize, shuffle=True
    )

    test_set = torch.load("data/processed/test_mnist.pt")
    
    testloader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batchsize, shuffle=True
    )

    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    epochs = args.epochs
    step = 0
    train_losses, test_losses, accuracies = [], [], []
    for e in range(args.epochs):
        running_loss = 0
        for images, labels in trainloader:
            optimizer.zero_grad()

            out = model(images)

            loss = criterion(out, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            step += 1
            if step % 100 == 0:
                os.makedirs("models/", exist_ok=True)
                torch.save(model.state_dict(), "models/trained_model.pt")
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
            epoch_loss = running_loss/len(trainloader)
            epoch_val_loss = running_val_loss/len(testloader)
            epoch_val_acc = running_accuracy/len(testloader)
            
            train_losses.append(epoch_loss)
            test_losses.append(epoch_val_loss)
            accuracies.append(epoch_val_acc)
            
            print(f"Testset accuracy: {epoch_val_acc}%")
            print(f"Validation loss: {epoch_val_loss}")
            print(f"Training loss: {epoch_loss}")

    plt.plot(np.arange(epochs), train_losses, label='training loss')
    plt.plot(np.arange(epochs), test_losses, label='validation loss')
    plt.plot(np.arange(epochs), accuracies, label='accuracy')
    plt.xlabel('epochs')
    plt.legend()
    plt.title('MNIST model training')
    
    os.makedirs("reports/figures/", exist_ok=True)
    plt.savefig("reports/figures/train_loss.png")
    


if __name__ == "__main__":
    train()
