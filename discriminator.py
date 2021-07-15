import torch
import torch.nn as nn
import numpy as np


class Discriminator(nn.Module):
    def __init__(self, max_len, nchars):
        super(Discriminator, self).__init__()
        self.FC_linear_1 = nn.Linear(max_len, max_len*2)
        self.FC_linear_2 = nn.Linear(max_len*2, (max_len*2)//2)
        self.FC_linear_3 = nn.Linear(
            (max_len*2)//2, (max_len*2)//8)
        self.FC_linear_4 = nn.Linear((max_len*2)//8, 1)
        self.LearkyRelu = nn.LeakyReLU(0.2)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        inputs = inputs.type(torch.float32)

        # print("-"*5, "Discriminator Start", "-"*5)
        inputs = inputs.reshape(inputs.shape[0], np.prod(inputs.shape[1:]))
        h = self.LearkyRelu(self.FC_linear_1(inputs))
        # print("after self.LearkyRelu(self.FC_linear_1(inputs))", h.shape)
        h = self.LearkyRelu(self.FC_linear_2(h))
        # print("after self.LearkyRelu(self.FC_linear_2(inputs))", h.shape)
        h = self.LearkyRelu(self.FC_linear_3(h))
        # print("after self.LearkyRelu(self.FC_linear_3(h))", h.shape)
        h = self.Sigmoid(self.FC_linear_4(h))
        # print("after self.LearkyRelu(self.FC_linear_4(h))", h.shape)
        # print("-"*5, "Discriminator End", "-"*5)
        return h
