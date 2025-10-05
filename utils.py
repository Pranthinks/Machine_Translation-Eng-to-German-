import torch
import torch.nn as nn
from torch.utils.data import dataloader, dataset, TensorDataset
import torch.nn.functional as F
import math

#My custom Positional Encodings
class Positional_Encoding(nn.Module):  # Fixed typo: "Postional" → "Positional"
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                            (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# My Custom Masking Function
class Masked_Attention(nn.Module):
    def __init__(self, num_heads, embedings, dropout = 0.1):
        super().__init__()
        assert embedings % num_heads == 0
        self.num_heads = num_heads
        self.embedings = embedings
        self.heads = embedings // num_heads  
        self.dropout = nn.Dropout(dropout) 

        self.fc1 = nn.Linear(embedings, embedings)
        self.fc2 = nn.Linear(embedings, embedings)
        self.fc3 = nn.Linear(embedings, embedings)
        self.outlayer = nn.Linear(embedings, embedings)
    
    def forward(self, x):
        batch_size, seq_len , embed = x.size()
        Q = self.fc1(x)
        K = self.fc2(x)
        V = self.fc3(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.heads).transpose(1, 2)
        
        first_term = torch.matmul(Q, K.transpose(-2, -1))
        second_term = math.sqrt(self.heads)
        scores = first_term/second_term
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        scores = scores.masked_fill(mask.to(scores.device), float('-inf'))

        val = F.softmax(scores, dim=-1)
        val = self.dropout(val)
        output = torch.matmul(val, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed)

        out = self.outlayer(output)
        return out

# My Custom Cross Attention Function
class Multihead(nn.Module):
    def __init__(self, num_heads, embeding, dropout = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embeding = embeding
        self.head = self.embeding // num_heads
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(embeding, embeding)
        self.fc2 = nn.Linear(embeding, embeding)
        self.fc3 = nn.Linear(embeding, embeding)
        self.out_layer = nn.Linear(embeding, embeding)
    
    def forward(self, x, enc_output=None):
        batch_size, seq_len, embed = x.size()
        Q = self.fc1(x)
        if enc_output is not None:
            K = self.fc2(enc_output)
            V = self.fc3(enc_output)
            seq_len_kv = enc_output.size(1)
        else:
            K = self.fc2(x)
            V = self.fc3(x)
            seq_len_kv = seq_len

        # Split heads
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)

        # Combine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, embed)
        return self.out_layer(out)

# My Custom Postional Feed Forward Network
class Position_Feedforward(nn.Module):
    def __init__(self, input_ch, output_ch, dropout = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(input_ch, output_ch)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(output_ch, input_ch)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self, x):
         x= self.fc1(x)
         x = self.relu(x)
         x = self.dropout(x)
         x = self.fc2(x)

         return x