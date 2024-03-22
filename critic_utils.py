import torch
import torch.nn as nn
# import gym
from typing import List, Dict

class CriticNet(nn.Module):
    def __init__(self,
                 feature_extractor,
                 hidden_sizes,
                 hidden_act=nn.ReLU):
        super().__init__()
        self.feature_extractor = feature_extractor
        in_size = feature_extractor.features_dim
        mlp_extractor : List[nn.Module] = []
        for curr_layer_dim in hidden_sizes:
            mlp_extractor.append(nn.Linear(in_size, curr_layer_dim))
            mlp_extractor.append(hidden_act())
            in_size = curr_layer_dim

        mlp_extractor.append(nn.Linear(in_size, 1))
        self.critic_net = nn.Sequential(*mlp_extractor)

    def forward(self, x: Dict[str, torch.Tensor]):
        feature = self.feature_extractor.forward(x)
        value = self.critic_net.forward(feature)

        return value