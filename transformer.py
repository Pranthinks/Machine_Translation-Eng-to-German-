import torch
import torch.nn as nn
from torch.utils.data import dataset, TensorDataset, dataloader
import torch.nn.functional as F
import math
from utils import Positional_Encoding, Position_Feedforward, Masked_Attention, Multihead

#Desigining Multi-layered Encoder
class Multi_Encoder(nn.Module):
    def __init__(self, d_model, heads, hidden_lay, dropout = 0.1):
        super().__init__()
        #Encoder Stuff
        self.E_Multi = Multihead(heads, d_model, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.E_FeedFor = Position_Feedforward(d_model, hidden_lay)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # Doing Encoder Stuff 
        x1 = self.E_Multi(x)
        x = self.norm1(x + self.dropout(x1))

        x2 = self.E_FeedFor(x)
        enc_output = self.norm2(x + self.dropout(x2))
        return enc_output
    
class Multi_Decoder(nn.Module):
    def __init__(self, d_model, heads, hidden_lay, dropout = 0.1):
        super().__init__()
        #Decoder stuff
        self.D_Mask = Masked_Attention(heads, d_model, dropout)
        self.norm3 = nn.LayerNorm(d_model)
        self.Cross_att = Multihead(heads, d_model, dropout)
        self.norm4 = nn.LayerNorm(d_model)
        self.D_Feedfor = Position_Feedforward(d_model, hidden_lay)
        self.norm5 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output):
        x3 = self.D_Mask(x)
        x = self.norm3(x+ self.dropout(x3))
        
        x4 = self.Cross_att(x, enc_output)  
        x = self.norm4(x + self.dropout(x4))
        
        x5 = self.D_Feedfor(x)
        x = self.norm5(x + self.dropout(x5))
        return x

    

class FullTransformer_Custom(nn.Module):
    def __init__(self, vocab_size, heads, d_model, hidden_lay,seq_len, num_layers,dropout = 0.1):
        super().__init__()
        self.src_embedding = nn.Embedding(vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.pos = Positional_Encoding(seq_len, d_model)
        self.pos1 = Positional_Encoding(seq_len,d_model)

        # Create 6 encoder and 6 decoder layers
        self.encoder_layers = nn.ModuleList([
            Multi_Encoder(d_model, heads, hidden_lay, dropout)
            for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            Multi_Decoder(d_model, heads, hidden_lay, dropout)
            for _ in range(num_layers)
        ])

        self.out_layer = nn.Linear(d_model, vocab_size)
    
    def encoder(self, src):
        # Doing Encoder Stuff 
        x = self.src_embedding(src) * math.sqrt(self.d_model)
        x = self.pos(x)

        for layer in self.encoder_layers:
            x = layer(x)
        return x
    
    def decoder(self, tar, enc_output):
        # Doing the Decoder Stuff
        x = self.tgt_embedding(tar) * math.sqrt(self.d_model)
        x = self.pos1(x)
        for layer in self.decoder_layers:
            x = layer(x, enc_output)
        out = self.out_layer(x)
        return out

    
    def forward(self, src, tar):
        endcoder_output = self.encoder(src)
        final = self.decoder(tar, endcoder_output)

        return final
        

        
#Usage 
obj = FullTransformer_Custom(10, 2, 4, 4, 4, 6)
x = torch.randint(0,10, (2, 4))
y = torch.randint(0, 10, (2, 3))
output = obj(x, y)
print(output)
