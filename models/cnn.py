import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )

        self.fc = nn.Sequential(
            nn.Linear(32*8*8, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )

        
    def forward(self, x):
        x = self.cnn(x)        
        x = x.view(-1, 32*8*8)
        x = self.fc(x)
        return x