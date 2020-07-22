import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 8, 5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(8, 16, 5, padding=1, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)  
        )

        self.fc = nn.Sequential(
            nn.Linear(16*3*3, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )

        
    def forward(self, x):
        x = self.cnn(x)        
        x = x.view(-1, 16*3*3)
        x = self.fc(x)
        return x