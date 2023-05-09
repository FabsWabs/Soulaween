import torch
import torch.nn as nn
import torch.nn.functional as F

NetParameters = {
    'h_layer': 128
}

class PlaceStoneLinear(nn.Module):
    def __init__(self, obs_space, act_space):
        super(PlaceStoneLinear,self).__init__()
        obs_space = torch.from_numpy(obs_space)
        self.hidden = nn.Linear(torch.prod(obs_space), NetParameters['h_layer'])
        self.out = nn.Linear(NetParameters['h_layer'], act_space)
        
    def forward(self,x):
        x = F.gelu(self.hidden(x.flatten()))
        x = self.out(x)
        return x

class ChooseSetLinear(nn.Module):
    def __init__(self, obs_space, act_space):
        super(ChooseSetLinear,self).__init__()
        obs_space = torch.from_numpy(obs_space)
        self.hidden = nn.Linear(torch.prod(obs_space), NetParameters['h_layer'])
        self.out = nn.Linear(NetParameters['h_layer'], act_space)
        
    def forward(self,x):
        x = F.gelu(self.hidden(x.flatten()))
        x = self.out(x)
        return x
    
class TargetQLinear(nn.Module):
    def __init__(self, obs_space, act_space):
        super(TargetQLinear,self).__init__()
        obs_space = torch.from_numpy(obs_space)
        self.hidden = nn.Linear(torch.prod(obs_space), NetParameters['h_layer'])
        self.out = nn.Linear(NetParameters['h_layer'], act_space)
        
    def forward(self,x):
        x = F.gelu(self.hidden(x.flatten(1)))
        x = self.out(x)
        return x