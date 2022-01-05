import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from model import CNN, Linear

sys.path.append(os.path.abspath(".."))


def predict():
    parser = argparse.ArgumentParser(description="Training arguments")
    parser.add_argument("--model", default="linear", type=str)
    parser.add_argument("--data", default="data/processed/example_data.pt", type=str)
    parser.add_argument("--model_instance", default="models/trained_model.pt")
    args = parser.parse_args(sys.argv[1:])
    print(args)

    if args.model == "CNN":
        model = CNN()
    else:
        model = Linear()

    state_dict = torch.load(args.model_instance)
    model.load_state_dict(state_dict)
    model.eval()

    images = torch.torch.load(args.data)
    loader = torch.utils.data.DataLoader(images, batch_size=1, shuffle=True)
    predictions = []

    for item in loader:
        with torch.no_grad():
            prediction = model(item)

            predicted_class = np.argmax(prediction)

            predictions.append([item, predicted_class])

    # create plot of predictions
    figure = plt.figure(figsize=(10, 8))
    cols, rows = int(len(loader) / 2), 2
    for i in range(1, cols * rows + 1):
        img, pred = predictions[i - 1]
        figure.add_subplot(rows, cols, i)
        plt.title(pred)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    os.makedirs("reports/figures/", exist_ok=True)
    plt.savefig("reports/figures/predictions.png")


if __name__ == "__main__":
    predict()
