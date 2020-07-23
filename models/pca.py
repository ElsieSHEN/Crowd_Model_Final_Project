import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from sklearn.decomposition import PCA

class PCAModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(6, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        pca = PCA(n_components=6)
        x = x.reshape(x.shape[0], -1)
        pca.fit(x)
        x = pca.transform(x)
        x = torch.from_numpy(x).float()
        x = x.view(-1, 6)
        x = self.fc(x)
        return x