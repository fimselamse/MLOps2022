import argparse
import sys

import torch
from torch import nn, optim
from data import mnist
from model import MyAwesomeModel
import matplotlib.pyplot as plt
import numpy as np


class TrainOREvaluate(object):
    """ Helper class that will help launch class methods as commands
        from a single script
    """
    def __init__(self):
        parser = argparse.ArgumentParser(
            description="Script for either training or evaluating",
            usage="python main.py <command>"
        )
        parser.add_argument("command", help="Subcommand to run")
        args = parser.parse_args(sys.argv[1:2])
        if not hasattr(self, args.command):
            print('Unrecognized command')
            
            parser.print_help()
            exit(1)
        # use dispatch pattern to invoke method with same name
        getattr(self, args.command)()
    
    def train(self):
        print("Training day and night")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('--lr', default=0.01, type=float)
        parser.add_argument('--epochs', default=10, type=int)
        parser.add_argument('--batchsize', default=64, type=int)
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement training loop here
        model = MyAwesomeModel()
        train_set, _ = mnist()
        trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batchsize, shuffle=True)
        
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        step = 0
        train_loss = []
        for e in range(args.epochs):
            running_loss = 0
            for images, labels in trainloader:
                optimizer.zero_grad()
                                    
                out = model(images)
                
                loss = criterion(out, labels)
                loss.backward()
                
                optimizer.step()
                
                running_loss += loss.item()
                train_loss.append(loss.item())
                step+=1
            else:
                print(f'Training loss: {running_loss/len(trainloader)}')
                torch.save(model.state_dict(), 'checkpoint.pth')
        plt.plot(np.arange(step), train_loss)
        plt.show()
                        
    def evaluate(self):
        print("Evaluating until hitting the ceiling")
        parser = argparse.ArgumentParser(description='Training arguments')
        parser.add_argument('load_model_from', default="C:/MLOPS/dtu_mlops/s1_getting_started/exercise_files/final_exercise/checkpoint.pth")
        # add any additional argument that you want
        args = parser.parse_args(sys.argv[2:])
        print(args)
        
        # TODO: Implement evaluation logic here
        model = MyAwesomeModel()
        state_dict = torch.load('checkpoint.pth')
        model.load_state_dict(state_dict)
        model.eval()
        
        _, test_set = mnist()
        testloader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)
        
        criterion = nn.NLLLoss()
        
        accuracy = 0
        test_loss = 0
        for images, labels in testloader:
            
            out = model.forward(images)
            test_loss += criterion(out, labels).item()

            ## Calculating the accuracy 
            # Model's output is log-softmax, take exponential to get the probabilities
            ps = torch.exp(out)
            # Class with highest probability is our predicted class, compare with true label
            equality = (labels.data == ps.max(1)[1])
            # Accuracy is number of correct predictions divided by all predictions, just take the mean
            accuracy += equality.type_as(torch.FloatTensor()).mean()
        
        print(f'Accuracy: {accuracy/len(testloader)*100}%')
if __name__ == '__main__':
    TrainOREvaluate()
    
    
    
    
    
    
    
    
    