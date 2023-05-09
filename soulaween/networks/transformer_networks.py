import torch.nn as nn
import torch.nn.functional as F

NetParameter = {
    'nEmb': 32, # 256,
    'nFw': 64, # 512,
    'nAttnHead': 4,
    'nLayer': 2
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

class PlaceStoneTransformer(nn.Module):
    def __init__(self, obs_space, act_space):
        super(PlaceStoneTransformer,self).__init__()       
        self.encoder_block = EncoderBlock(**NetParameter, nInput=obs_space[0])
        self.out = nn.Linear(NetParameter['nEmb'] * 16, act_space)
        # self.out = nn.Conv1d(NetParameter['nEmb'] * 16, 32, 1)
        
    def forward(self,x):
        x = self.encoder_block(x.unsqueeze(0))
        x = self.out(x.flatten())
        return x

class ChooseSetTransformer(nn.Module):
    def __init__(self, obs_space, act_space):
        super(ChooseSetTransformer,self).__init__()       
        self.encoder_block = EncoderBlock(**NetParameter, nInput=obs_space[0])
        self.out = nn.Linear(NetParameter['nEmb'] * 16, act_space)
        # self.out = nn.Conv1d(NetParameter['nEmb'] * 16, 10, 1)
        
    def forward(self,x):
        x = self.encoder_block(x.unsqueeze(0))
        x = self.out(x.flatten())
        return x


class TargetQTransformer(nn.Module):
    def __init__(self, obs_space, act_space):
        super(TargetQTransformer,self).__init__()       
        self.encoder_block = EncoderBlock(**NetParameter, nInput=obs_space[0])
        self.out = nn.Linear(NetParameter['nEmb'] * 16, act_space)
        # self.out = nn.Conv1d(NetParameter['nEmb'], 1, 1)
        
    def forward(self,x):       
        x = self.encoder_block(x)
        x = self.out(x.flatten(1))
        return x
