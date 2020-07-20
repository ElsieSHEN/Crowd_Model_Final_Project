import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.decomposition import PCA

class PCAModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        pca = PCA(n_components=6)
        x = x.reshape(6, -1)
        pca.fit(x)
        pca.transform(x)
        x = x.view(-1, 3*32*32)
        x = self.fc(x)
        return x