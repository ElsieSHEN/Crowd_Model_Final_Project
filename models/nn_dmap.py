import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

from datafold.dynfold import DiffusionMaps
import datafold.pcfold as pfold

class CNN_DMap(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        #print(x.shape)
        x = x.view(500, -1)
        X_pcm = pfold.PCManifold(x)
        X_pcm.optimize_parameters(result_scaling=2)
        dmap = DiffusionMaps(kernel=pfold.GaussianKernel(epsilon=X_pcm.kernel.epsilon), 
                     n_eigenpairs=6, dist_kwargs=dict(cut_off=X_pcm.cut_off))

        dmap = dmap.fit(X_pcm)
        dmap = dmap.set_coords([1, 2])
        X_pcm = dmap.transform(X_pcm)
        X_pcm = torch.from_numpy(X_pcm).float()
        #print(X_pcm.shape)

        x = X_pcm.view(-1, 2)
        x = self.fc(x)
        return x