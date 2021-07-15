import torch
import torch.nn as nn
import numpy as np


class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, max_len):
        super(Generator, self).__init__()
        self.max_len = max_len
        self.hidden_dim = hidden_dim

        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.FC_linear1 = nn.Linear(latent_dim, hidden_dim)
        self.batch1 = nn.BatchNorm1d(hidden_dim)
        self.FC_GRU1 = nn.GRU(max_len, hidden_dim)
        self.batch2 = nn.BatchNorm1d(hidden_dim)
        self.drop = nn.Dropout(0.3)
        self.FC_linear2 = nn.Linear(hidden_dim*hidden_dim, max_len)

    def forward(self, x):
        # print("-"*5, "Generator Start", "-"*5)
        h = self.LeakyReLU(self.FC_linear1(x))
        h = self.batch1(h)
        # print("after self.batch1(h)", h.shape)
        h = torch.repeat_interleave(h, self.max_len)
        # print("after torch.repeat_interleave(h, self.max_len)", h.shape)
        h = h.reshape(-1, self.hidden_dim, self.max_len)
        h = self.LeakyReLU(self.FC_GRU1(h)[0])
        h = self.batch2(h)
        h = h.reshape(-1, np.prod(h.shape[1:]))
        x_hat = torch.softmax(self.FC_linear2(h), dim=-1)

        return x_hat
