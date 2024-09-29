import torch.nn as nn
import torch
import numpy as np
from .util import init

"""MLP modules.
   input: 1*5  output: 1*64
"""

class PhiLayer(nn.Module):
    def __init__(self, input_dim, use_orthogonal, use_ReLU):
        super(PhiLayer, self).__init__()

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.phi = nn.Sequential(
            init_(nn.Linear(input_dim, 64)), active_func, nn.LayerNorm(64))  # 输入5， 输出64
        
    def forward(self, x):
        x = self.phi(x)        
        return x


class PhiNetBase(nn.Module):
    def __init__(self, args, input_dim, cat_self=True, attn_internal=False):
        super(PhiNetBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(input_dim)

        self.phi_net = PhiLayer(input_dim, self._use_orthogonal, self._use_ReLU)

    def forward(self, x):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.phi_net(x)

        return x