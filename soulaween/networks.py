import torch.nn as nn
import torch.nn.functional as F

NetParameter = {
    'nInput':9,
    'nEmb': 64, # 256,
    'nFw': 128, # 512,
    'nAttnHead': 4,
    'nLayer': 3
}


class EncoderBlock(nn.Module):
    def __init__(self, nInput, nEmb, nFw, nAttnHead, nLayer):
        super(EncoderBlock,self).__init__()
        self.f1 = nn.Conv1d(nInput, nFw, 1)
        self.f2 = nn.Conv1d(nFw, nEmb, 1)
        attn_layer = nn.TransformerEncoderLayer(nEmb, nAttnHead, nFw)
        self.attn_encoder = nn.TransformerEncoder(attn_layer, nLayer)
        
    def forward(self,x): 
        x = self.f2(F.relu(self.f1(x)))
        x = F.layer_norm(x,[x.size(-1)])
        x = x.permute(2,0,1)
        x = self.attn_encoder(x)
        x = x.permute(1,2,0)  
        return x

class PlaceStoneNet(nn.Module):
    def __init__(self, act_space):
        super(PlaceStoneNet,self).__init__()       
        self.encoder_block = EncoderBlock(**NetParameter)
        self.out = nn.Linear(NetParameter['nEmb'] * 16, act_space)
        # self.out = nn.Conv1d(NetParameter['nEmb'] * 16, 32, 1)
        
    def forward(self,x):
        x = self.encoder_block(x.unsqueeze(0))
        x = self.out(x.flatten())
        return x

class ChooseSetNet(nn.Module):
    def __init__(self, act_space):
        super(ChooseSetNet,self).__init__()       
        self.encoder_block = EncoderBlock(**NetParameter)
        self.out = nn.Linear(NetParameter['nEmb'] * 16, act_space)
        # self.out = nn.Conv1d(NetParameter['nEmb'] * 16, 10, 1)
        
    def forward(self,x):
        x = self.encoder_block(x.unsqueeze(0))
        x = self.out(x.flatten())
        return x


class TargetQNet(nn.Module):
    def __init__(self, act_space):
        super(TargetQNet,self).__init__()       
        self.encoder_block = EncoderBlock(**NetParameter)
        self.out = nn.Linear(NetParameter['nEmb'] * 16, act_space)
        # self.out = nn.Conv1d(NetParameter['nEmb'], 1, 1)
        
    def forward(self,x):       
        x = self.encoder_block(x)
        x = self.out(x.flatten(1))
        return x
