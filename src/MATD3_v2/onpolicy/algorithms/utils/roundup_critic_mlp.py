import torch.nn as nn
import torch
import numpy as np
from .util import init, get_clones

"""MLP modules.
   input: 24*N   output: V(s)
"""

class C_MLPLayer(nn.Module):
    def __init__(self, input_dim, hidden_size, layer_N, use_orthogonal, use_ReLU, use_centralized_V):
        super(C_MLPLayer, self).__init__()
        self._layer_N = layer_N
        self.use_centralized_V = use_centralized_V
        N = int(input_dim / 24)

        active_func = [nn.Tanh(), nn.ReLU()][use_ReLU]
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]
        gain = nn.init.calculate_gain(['tanh', 'relu'][use_ReLU])

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=gain)


        #self.phi_loc = nn.Sequential(
        #    init_(nn.Linear(6, 64)), active_func, nn.LayerNorm(64))  # 输入6， 输出64

        #self.phi_oij = nn.Sequential(
        #    init_(nn.Linear(5, 64)), active_func, nn.LayerNorm(64))  # 输入5， 输出64

        action_dim = 2 * N if self.use_centralized_V else 2
        self.fc1 = nn.Sequential(
            init_(nn.Linear(input_dim+action_dim, 256)), active_func, nn.LayerNorm(256))

        self.fc2 = nn.Sequential(
            init_(nn.Linear(256, 128)), active_func, nn.LayerNorm(128))

        self.fc3 = nn.Sequential(
            init_(nn.Linear(128, 32)), active_func, nn.LayerNorm(32))

    def forward(self, x, actions):
        len_x = x.size(1)  # col. the dim of x
        N = int(len_x / 24)  # num of agents

        if self.use_centralized_V:
            actions = actions.reshape(actions.size(0) // N, -1)
            actions = torch.unsqueeze(actions, 1).repeat_interleave(N, axis=1)
            actions = actions.reshape(actions.size(0) * N, -1)

        x = torch.cat((x, actions), dim=1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


class C_MLPBase(nn.Module):
    def __init__(self, args, obs_shape, cat_self=True, attn_internal=False):
        super(C_MLPBase, self).__init__()

        self._use_feature_normalization = args.use_feature_normalization
        self._use_orthogonal = args.use_orthogonal
        self._use_ReLU = args.use_ReLU
        self._stacked_frames = args.stacked_frames
        self._layer_N = args.layer_N
        self.hidden_size = args.hidden_size
        self.use_centralized_V = args.use_centralized_V

        obs_dim = obs_shape[0]

        if self._use_feature_normalization:
            self.feature_norm = nn.LayerNorm(obs_dim)

        self.mlp = C_MLPLayer(obs_dim, self.hidden_size,
                              self._layer_N, self._use_orthogonal, self._use_ReLU, self.use_centralized_V)

    def forward(self, x, actions):
        if self._use_feature_normalization:
            x = self.feature_norm(x)

        x = self.mlp(x, actions)

        return x
