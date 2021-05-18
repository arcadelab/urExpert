import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

def clones_layers(module, n):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequentMask = np.triu(np.ones(attn_shape), k=1).astype(np.bool_)
    return torch.from_numpy(subsequentMask) == 0

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, 1e-9)
    p_atten = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_atten = dropout(p_atten)
    return torch.matmul(p_atten, value), p_atten

class MultiHeadAttention(nn.Module):
    """docstring for MultiHeadAttention"""
    def __init__(self, h, input_size, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert input_size % h == 0
        self.d_k = input_size // h
        self.h = h
        self.linears = clones_layers(nn.Linear(input_size, input_size), 4)
        self.atten = None
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key,value, mask=mask, dropout=self.dropout)
        x =  x.transpose(1,2).contiguous().view(nbatches, -1, self.h*self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    def __init__(self, input_size, hidden_layer, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(input_size, hidden_layer)
        self.w_2 = nn.Linear(hidden_layer, input_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embedding(nn.Module):
    """docstring for Embedding for integer input"""
    def __init__(self, input_size, vocab):
        super(Embedding, self).__init__()
        self.lut = nn.Embedding(vocab, input_size)
        self.input_size = input_size

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.input_size)

class Linear(nn.Module):
    """docstring for Linear embbeding for float input"""
    def __init__(self, input_size, vocab):
        super(Linear, self).__init__()
        self.lut = nn.Linear(vocab, input_size)
        self.input_size = input_size

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.input_size)

class PositionalEncoding(nn.Module):
    def __init__(self, input_size, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, input_size)
        positoin = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, input_size, 2) * -(math.log(10000.0) / input_size))
        pe[:, 0::2] = torch.sin(positoin * div_term)
        pe[:, 1::2] = torch.cos(positoin * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        with torch.no_grad():
            pe = self.pe[:, :x.size(1)].clone()
        return self.dropout(x + pe)        

class EncoderDecoder(nn.Module):
    """standard code for EncoderDecoder"""
    def __init__(self, src_vocab, tgt_vocab, N=6, input_size=512, hidden_layer=2048, h=8, dropout=0.1, task_dim=None):
        '''
        src_vocab: source vocab length which corresponding to the dimension of kinematics 
        tgt_vocab, target vocab length which corresponding to the dimension of kinematics 
        N: Number of encoder and decoder layers
        Input_size: dimension of the intermediate layers
        hidden_layer: the hidden_layer dimenson of the positionwise ff network
        h: number of heads for multi head attention
        dropout: dropout ratio for dropout layer
        task_dim: if None means no task series will be input otherwise means dimension of the task input
        '''
        super(EncoderDecoder, self).__init__()
        c = copy.deepcopy
        attn = MultiHeadAttention(h, input_size, dropout)
        ff = PositionwiseFeedForward(input_size, hidden_layer, dropout)
        positoin = PositionalEncoding(input_size, dropout)
        src_task_embeded = None
        real_input_size = input_size
        if task_dim is not None:
            self.src_task_embeded = nn.Sequential(Embedding(input_size, task_dim), c(positoin))
            real_input_size += input_size
            self.src_dim_reduce = nn.Linear(real_input_size, input_size)
        self.encoder=Encoder(EncoderLayer(input_size, c(attn), c(ff), dropout), N)
        self.decoder=Decoder(DecoderLayer(input_size, c(attn), c(attn), c(ff), dropout), N)
        self.src_embeded=nn.Sequential(Linear(input_size, src_vocab), c(positoin))
        self.tgt_embeded=nn.Sequential(Linear(input_size, tgt_vocab), c(positoin))
        self.generator=Generator([input_size], tgt_vocab)

        for p in self.parameters():
            if p.dim() > 1:
                 nn.init.xavier_uniform_(p)

    def forward(self, src, tgt, src_mask, tgt_mask, src_task=None):
        src_in = src - src[:,0].unsqueeze(dim=1)
        tgt_in = tgt - src[:,0].unsqueeze(dim=1)
        return self.decode(self.encode(src_in, src_mask, src_task), src_mask, tgt_in, tgt_mask) + src[:, 0, :].unsqueeze(dim=1)

    def encode(self, src, src_mask, src_task=None):
        embedding = self.src_embeded(src)
        if src_task is not None:
            embedding = torch.cat([embedding, self.src_task_embeded(src_task)], dim=2)
            embedding = self.src_dim_reduce(embedding)
        return self.encoder(embedding, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.generator(self.decoder(self.tgt_embeded(tgt), memory, src_mask, tgt_mask))

    def generate(self, src, src_mask, target_length):
        memory = self.encode(src, src_mask)
        tgt = src.new_zeros(src.shape[0], 1, src.shape[2])
        tgt[:,0,:] = src[:,-1,:]
        for i in range(target_length):
            pred = self.decode(memory, src_mask, tgt, subsequent_mask(tgt.size(1)).type_as(tgt))
            tgt = torch.cat([pred[:,-1,:].unsqueeze(1), tgt], dim=1)
        return tgt[:,1:,:]


class Generator(nn.Module):
    """
    standard linear predictor consists of several linear layers
    """
    def __init__(self, layer_dims, output_dim):
        super(Generator, self).__init__()
        self.layers = []
        assert len(layer_dims) >= 1
        for i in range(len(layer_dims)-1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
        self.pred = nn.Linear(layer_dims[-1], output_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.pred(x)

class Encoder(nn.Module):
    #Encoder layer consists of several encoder layer and has layer norm at the end
    def __init__(self, layer, n):
        super(Encoder, self).__init__()
        self.layers = clones_layers(layer, n)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a = nn.Parameter(torch.ones(features))
        self.b = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a * (x - mean) / (std + self.eps) + self.b

class SublayerConnection(nn.Module):
    """docstring for SublayerConnection"""
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderLayer(nn.Module):
    """docstring for EncoderLayer: consists of self attention, feed forward and sublayer connections"""
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayers =  clones_layers(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayers[1](x, self.feed_forward)

class Decoder(nn.Module):
    """docstring for Decoder"""
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones_layers(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    """Decpde layer consists of self ateention, cross attention, feed forward and sublayer connections"""
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayers =  clones_layers(SublayerConnection(size, dropout), 3)
        self.size = size

    def forward(self, x, memory, src_mask, tgt_mask):
        m = memory
        x = self.sublayers[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayers[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayers[2](x, self.feed_forward)


















        
        