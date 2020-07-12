import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

def train(model, trainloader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    epochs = 10

    print('Start training!')

    for e in range(epochs):
        train_loss = 0
        
        for data, target in trainloader:
            optimizer.zero_grad()
            
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * data.size(0)
            
        train_loss = train_loss/len(trainloader.dataset)
        
        print('Epoch: {} \t Training Loss:{:.6f}'.format(e+1, train_loss))
        
    print('Finish training!')