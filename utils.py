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

class Masked_Attention(nn.Module):
    """Improved masked self-attention with efficient causal masking"""
    def __init__(self, num_heads, embedings, dropout=0.1):
        super().__init__()
        assert embedings % num_heads == 0, "embedings must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.embedings = embedings
        self.heads = embedings // num_heads
        self.dropout = nn.Dropout(dropout)
        
        # QKV projections (keeping your original names)
        self.fc1 = nn.Linear(embedings, embedings)
        self.fc2 = nn.Linear(embedings, embedings)
        self.fc3 = nn.Linear(embedings, embedings)
        self.outlayer = nn.Linear(embedings, embedings)
        
        # Register causal mask as buffer (moved to device automatically)
        self.register_buffer('causal_mask', None)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Project to Q, K, V
        Q = self.fc_q(x)
        K = self.fc_k(x)
        V = self.fc_v(x)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply padding mask if provided
        if mask is not None:
            padding_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        # Apply causal mask (prevents attending to future tokens)
        if self.causal_mask is None or self.causal_mask.size(0) != seq_len:
            # Create upper triangular matrix (1s above diagonal)
            self.causal_mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device),
                diagonal=1
            ).bool()
        
        scores = scores.masked_fill(self.causal_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Reshape back to [B, S, E]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final linear projection
        output = self.out_layer(output)
        return output


class Multihead_Attention(nn.Module):
    """Improved multi-head attention for encoder and cross-attention"""
    def __init__(self, num_heads, embeddings, dropout=0.1):
        super().__init__()
        assert embeddings % num_heads == 0, "embeddings must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.embeddings = embeddings
        self.head_dim = embeddings // num_heads
        self.scale = math.sqrt(self.head_dim)
        
        # QKV projections
        self.fc_q = nn.Linear(embeddings, embeddings)
        self.fc_k = nn.Linear(embeddings, embeddings)
        self.fc_v = nn.Linear(embeddings, embeddings)
        self.out_layer = nn.Linear(embeddings, embeddings)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, enc_output=None, mask=None):
        batch_size, seq_len, embed_dim = x.size()
        
        # Query from decoder
        Q = self.fc_q(x)
        
        # Key and Value from encoder (if cross-attention) or from x (self-attention)
        if enc_output is not None:
            K = self.fc_k(enc_output)
            V = self.fc_v(enc_output)
            seq_len_kv = enc_output.size(1)
        else:
            K = self.fc_k(x)
            V = self.fc_v(x)
            seq_len_kv = seq_len
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len_kv, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply padding mask if provided
        if mask is not None:
            padding_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, S]
            scores = scores.masked_fill(padding_mask, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, V)
        
        # Reshape back to [B, S, E]
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        
        # Final linear projection
        output = self.out_layer(output)
        return output

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