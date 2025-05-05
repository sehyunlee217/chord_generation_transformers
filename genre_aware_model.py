import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super(InputEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
    
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)

        # create matrix of (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        position_idx = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position_idx * div_term)
        pe[:, 1::2] = torch.cos(position_idx * div_term)
        
        # Modified to work with (batch_size, seq_len, d_model) format
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x has shape (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.layer_norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)
    
class FFN(nn.Module):
    def __init__(self, d_model, ffn_size, dropout=0.1):
        super(FFN, self).__init__()
        self.linear1 = nn.Linear(d_model, ffn_size)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(ffn_size, d_model)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        assert (
            self.d_k * num_heads == d_model
        ), "Embedding size needs to be divisible by heads"

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(num_heads * self.d_k, d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask=None, dropout=None):
        d_k = query.shape[-1]

        # query: (batch_size, num_heads, seq_len, d_k)
        # key: (batch_size, num_heads, seq_len, d_k)
        # key.transpose(-2, -1): (batch_size, num_heads, d_k, seq_len)
        scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k) # (batch_size, num_heads, seq_len, d_k) @ (batch_size, num_heads, d_k, seq_len) --> (batch_size, num_heads, seq_len, seq_len)
        if mask is not None:
            # mask: (batch_size, 1, 1, seq_len)
            scores = scores.masked_fill(mask == 0, -1e9) # --> if mask == 0, set score to -inf
        # scores = (batch_size, num_heads, n_row, n_col)
        scores = F.softmax(scores, dim=-1)

        # e.g) 
        # scores = torch.tensor([
        #     [1.0, 2.0, 3.0],   # Query 1 attends to 3 keys
        #     [1.0, 2.0, 4.0]    # Query 2 attends to 3 keys
        # ])
        # softmaxed = F.softmax(scores, dim=-1)
        # print(softmaxed)

        #     tensor([
        # [0.0900, 0.2447, 0.6652],  # sum = 1
        # [0.0420, 0.1142, 0.8438]   # sum = 1
        # ])

        if dropout is not None:
            scores = dropout(scores)
        
        return scores @ value, scores # (batch_size, num_heads, seq_len, d_k) @ (batch_size, num_heads, seq_len, d_k) --> (batch_size, num_heads, seq_len, d_k)

    def forward(self, q, k, v, mask=None): # mask: if we want some words to not interact with others
        query = self.W_q(q) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        key = self.W_k(k) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        value = self.W_v(v) # (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

        # (batch_size, num_heads, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2) 
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)

        x, self.attn = self.attention(query, key, value, mask=mask, dropout=self.dropout) 

        # (batch_size, num_heads, seq_len, d_k) --> (batch_size, seq_len, num_heads, d_k)
        x = x.transpose(1,2) # swap 1 and 2 dimensions

        # (batch_size, seq_len, num_heads * d_k) --> (batch_size, seq_len, d_model)
        x = x.contiguous().view(x.shape[0], x.shape[1], self.num_heads * self.d_k)

        #Difference between transpose vs view

        return self.W_o(x) # (batch_size, seq_len, d_model)
    

class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super(ResidualConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer): # ---> pass in sublayer as a function
        return x + self.dropout(sublayer(self.norm(x))) # (batch_size, seq_len, d_model) + (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)


class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, ffn: FFN, d_model, num_heads, dropout=0.1):
        super(EncoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.ffn = ffn
        self.residual_connection1 = ResidualConnection(d_model, dropout)
        self.residual_connection2 = ResidualConnection(d_model, dropout)

    def forward(self, x, mask=None):
        x = self.residual_connection1(x, lambda x: self.self_attention_block(x, x, x, mask=mask))
        x = self.residual_connection2(x, self.ffn)
        return x
    
class Encoder(nn.Module):
    def __init__(self, layers):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm(self.layers[0].self_attention_block.d_model)
    
    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return self.norm(x)
    
class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, ffn: FFN, d_model, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.ffn = ffn
        self.residual_connection1 = ResidualConnection(d_model, dropout)
        self.residual_connection2 = ResidualConnection(d_model, dropout)
        self.residual_connection3 = ResidualConnection(d_model, dropout)

    # x: input of decoder, enc_output: output of encoder
    def forward(self, x, enc_output, src_mask=None, tgt_mask=None): # src_mask: mask for encoder, tgt_mask: mask for decoder
        # self attention, K, Q, V = x, x, x
        x = self.residual_connection1(x, lambda x: self.self_attention_block(x, x, x, mask=tgt_mask))
        # cross attention, Q -> decoder, K, V -> encoder
        x = self.residual_connection2(x, lambda x: self.cross_attention_block(x, enc_output, enc_output, mask=src_mask)) 
        x = self.residual_connection3(x, self.ffn)
        return x
    
class Decoder(nn.Module):
    def __init__(self, layers):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = LayerNorm(self.layers[0].self_attention_block.d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.norm(x)
    
class Projector(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Projector, self).__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.linear(x), dim=-1)
    
class GenreEmbedding(nn.Module):
    def __init__(self, len_genre, d_model):
        super(GenreEmbedding, self).__init__()
        self.embedding = nn.Embedding(len_genre, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
    
class Transformer(nn.Module):
    def __init__(self, encoder, decoder, projector, src_embed, tgt_embed, genre_embed, src_pos_enc, tgt_pos_enc):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.projector = projector
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.genre_embed = genre_embed
        self.src_pos_enc = src_pos_enc
        self.tgt_pos_enc = tgt_pos_enc
    
    def encode(self, src, genre, src_mask=None):
        # source embedding
        x = self.src_embed(src) # (batch_size, seq_len, d_model)

        # genre embedding
        genre = self.genre_embed(genre)
        genre = genre.unsqueeze(1)  # (batch_size, 1, d_model)
        genre = genre.expand(-1, x.size(1), -1) # (batch_size, seq_len, d_model)

        # add genre embedding to source embedding
        x = x + genre # (batch_size, seq_len, d_model) + (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)
        x = self.src_pos_enc(x)
        x = self.encoder(x, mask=src_mask)
        return x
    
    def decode(self, tgt, enc_output, genre, src_mask=None, tgt_mask=None):
        x = self.tgt_embed(tgt)

        # genre embedding
        genre = self.genre_embed(genre)
        genre = genre.unsqueeze(1)  # (batch_size, 1, d_model)
        genre = genre.expand(-1, x.size(1), -1) # (batch_size, seq_len, d_model)

        # add genre embedding to target embedding
        x = x + genre # (batch_size, seq_len, d_model) + (batch_size, seq_len, d_model) --> (batch_size, seq_len, d_model)

        x = self.tgt_pos_enc(x)
        x = self.decoder(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return x
    
    def project(self, x):
        return self.projector(x)
    
    
def build_transformer(src_vocab_size, tgt_vocab_size, src_seq_len, tgt_seq_len, genre_len, d_model: int = 512, N: int = 6, num_heads: int = 8, ffn_size: int = 2048, dropout: float = 0.1):
    # embeddings & positional encodings
    src_embed = InputEmbedding(src_vocab_size, d_model)
    tgt_embed = InputEmbedding(tgt_vocab_size, d_model)
    genre_embed = GenreEmbedding(genre_len, d_model)  
    src_pos_enc = PositionalEncoding(d_model, max_len=src_seq_len)
    tgt_pos_enc = PositionalEncoding(d_model, max_len=tgt_seq_len)

    # encoder blocks
    encoder_layers = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        ffn = FFN(d_model, ffn_size, dropout=dropout)
        encoder_layers.append(EncoderBlock(self_attention_block, ffn, d_model, num_heads, dropout=dropout))

    # decoder blocks
    decoder_layers = []
    for _ in range(N):
        self_attention_block = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        cross_attention_block = MultiHeadAttention(d_model, num_heads, dropout=dropout)
        ffn = FFN(d_model, ffn_size, dropout=dropout)
        decoder_layers.append(DecoderBlock(self_attention_block, cross_attention_block, ffn, d_model, dropout=dropout))

    # encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_layers))
    decoder = Decoder(nn.ModuleList(decoder_layers))

    # projector
    projector = Projector(d_model, tgt_vocab_size)
    transformer = Transformer(encoder, decoder, projector, src_embed, tgt_embed, genre_embed, src_pos_enc, tgt_pos_enc)

    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.constant_(p, 0.1)

    return transformer