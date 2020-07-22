import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from megaman.geometry import Geometry
from megaman.embedding import (Isomap, LocallyLinearEmbedding,
                               LTSA, SpectralEmbedding)

class Mega_LTSA_Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(3*32*32, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        
        rad1 = 0.7272        
        radius = rad1
        adjacency_method = 'cyflann'
        adjacency_kwds = {'radius':radius}
        affinity_method = 'gaussian'
        affinity_kwds = {'radius':radius}
        laplacian_method = 'symmetricnormalized'
        laplacian_kwds = {'scaling_epps':radius}
        
        geom = Geometry(adjacency_method=adjacency_method, adjacency_kwds=adjacency_kwds,
                affinity_method=affinity_method, affinity_kwds=affinity_kwds,
                laplacian_method=laplacian_method, laplacian_kwds=laplacian_kwds)
        x = x.view(-1, 3*32*32)
        #print(x.shape)
        geom.set_data_matrix(x)

        ltsa = LTSA(n_components=2,eigen_solver='dense',geom=geom)
        X_ltsa = ltsa.fit_transform(x)

        X_ltsa = X_ltsa.view(-1, 3*32*32)
        X_ltsa = self.fc(X_ltsa)
        return X_ltsa