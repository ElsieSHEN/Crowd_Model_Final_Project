import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

class FCModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        x = x.view(-1, 3*32*32)
        x = self.fc(x)
        return x