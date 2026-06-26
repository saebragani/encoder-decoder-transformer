import torch
import torch.nn as nn
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model) # Lookup table

    def forward(self, x):
        x = self.token_embedding(x) # (B, seq_len) --> (B, seq_len, d_model)
        x = x * math.sqrt(self.d_model) #  scale the embeddings before adding positional encoding to keep the two on a similar magnitude
        return x


class PsitionEncoding(nn.Module):
    def __init__(self, seq_len, d_model, dropout):
        super().__init__()
        position_encoding = torch.zeros(seq_len, d_model)
        
        pos = torch.arange(0, seq_len).unsqueeze(-1) # --> (seq_len, 1)
        i = torch.arange(0, d_model, 2).unsqueeze(0) # --> (1, d_model//2)
        position_encoding[:, 0::2] = torch.sin(pos/10000**(i/d_model))
        position_encoding[:, 1::2] = torch.cos(pos/10000**(i/d_model))
        
        # need to register as buffer to let Pytorch not train the weights
        self.register_buffer("position_encoding", position_encoding) # (seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.position_encoding.unsqueeze(0)[:, :x.shape[1], :] # (B, seq_len, d_model) + (1, seq_len, d_model) --> Pytorch broadcasting takes care of batch dimension
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model should be divisible by h"
        self.d_k = d_model // h
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        
        self.w_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(q, k, v, dropout: nn.Dropout, mask):
        d_k = q.shape[-1]
        attention_scores = q @ k.transpose(-2, -1) / math.sqrt(d_k) # skippin B, h (seq_len, d_k) @ (d_k, seq_len) --> (seq_len, seq_len) or (B, h, seq_len, seq_len)
        if mask is not None:
            attention_scores = attention_scores.masked_fill_(mask==0, float("-inf"))
        
        attention_scores = attention_scores.softmax(dim=-1)
        attention_scores = dropout(attention_scores)
        x = attention_scores @ v # (seq_len, seq_len) @ (seq_len, d_k) --> (seq_len, d_k) or (B, h, seq_len, d_k)
        return x, attention_scores

    def forward(self, q, k, v, mask=None): # x --> (B, seq_len, d_model)
        q = self.w_q(q) # --> (B, seq_len, d_model)
        k = self.w_k(k) # --> (B, seq_len, d_model)
        v = self.w_v(v) # --> (B, seq_len, d_model)

        q = q.view(q.shape[0], q.shape[1], self.h, self.d_k).transpose(1, 2) # (B, h, seq_len, d_k)
        k = k.view(k.shape[0], k.shape[1], self.h, self.d_k).transpose(1, 2) # (B, h, seq_len, d_k)
        v = v.view(v.shape[0], v.shape[1], self.h, self.d_k).transpose(1, 2) # (B, h, seq_len, d_k)

        x, self.attention_scores = MultiHeadAttention.attention(q, k, v, self.dropout, mask) # (B, h, seq_len, d_k)
        x = x.transpose(1, 2).contiguous() # (B, seq_len, h, d_k)
        x = x.view(x.shape[0], x.shape[1], int(self.h * self.d_k)) # (B, seq_len, d_model)
        
        x = self.w_o(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, d_model, dropout):
        super().__init__()
        self.l1 = nn.Linear(d_model, 4*d_model)
        self.ReLU = nn.ReLU()
        self.l2 = nn.Linear(4*d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.l1(x) # (B, seq_len, d_model) --> (B, seq_len, 4*d_model)
        x = self.ReLU(x)
        x = self.dropout(x)
        x = self.l2(x) # (B, seq_len, 4*d_model) --> (B, seq_len, d_model)
        return x


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(d_model)) # scale each column by one value (alpha)
        self.bias = nn.Parameter(torch.zeros(d_model)) # shift each column by one value (bias)

    def forward(self, x):
        # standardize values for each row; x (B, seq_len, d_model)
        x = (x - x.mean(dim=-1, keepdim=True)) / torch.sqrt(x.var(dim=-1, keepdim=True) + self.eps)
        # for row_id in range(x.shape[1]):
        #     row = x[:, row_id, :]
        #     row_mean = row.mean()
        #     row_var = row.std() ** 2
        #     x[:, row_id, :] = (row - row_mean) / math.sqrt(row_var + eps)

        x = x * self.alpha + self.bias
        return x


class ResidualConnection(nn.Module):
    def __init__(self, d_model, dropout, eps=1e-6):
        super().__init__()
        self.norm = LayerNorm(d_model, eps)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, sublayer: MultiHeadAttention | FeedForward):
        x = x + self.dropout(self.norm(sublayer(x)))
        return x


class EncoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = FeedForward(d_model, dropout)
        self.residual_connection1 = ResidualConnection(d_model, dropout)
        self.residual_connection2 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_mask=None):
        x = self.residual_connection1(x, lambda x: self.multi_head_attention(x, x, x, encoder_mask))
        x = self.residual_connection2(x, self.feed_forward)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, h, dropout):
        super().__init__()
        self.multi_head_attention1 = MultiHeadAttention(d_model, h, dropout)
        self.multi_head_attention2 = MultiHeadAttention(d_model, h, dropout)
        self.feed_forward = FeedForward(d_model, dropout)
        self.residual_connection1 = ResidualConnection(d_model, dropout)
        self.residual_connection2 = ResidualConnection(d_model, dropout)
        self.residual_connection3 = ResidualConnection(d_model, dropout)

    def forward(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.residual_connection1(x, lambda x: self.multi_head_attention1(x, x, x, decoder_mask))
        x = self.residual_connection2(x, lambda x: self.multi_head_attention2(x, encoder_output, encoder_output, encoder_mask))
        x = self.residual_connection3(x, self.feed_forward)
        return x


class ProjectionLayer(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        x = self.linear(x) # (B, seq_len, d_model) --> (B, seq_len, vocab_size)
        # Softmax is included in nn.CrossEntropyLoss
        return x


class TransformerBlock(nn.Module):
    def __init__(self, source_vocab_size, target_vocab_size, source_seq_len, target_seq_len, d_model, h, dropout, n_encoder, n_decoder):
        super().__init__()
        self.source_token_embedding = TokenEmbedding(source_vocab_size, d_model)
        self.source_psition_encoding = PsitionEncoding(source_seq_len, d_model, dropout)
        self.target_token_embedding = TokenEmbedding(target_vocab_size, d_model)
        self.target_psition_encoding = PsitionEncoding(target_seq_len, d_model, dropout)
        
        self.encoder_module_list = nn.ModuleList([EncoderBlock(d_model, h, dropout) for _ in range(n_encoder)])
        self.decoder_module_list = nn.ModuleList([DecoderBlock(d_model, h, dropout) for _ in range(n_decoder)])
        self.projection = ProjectionLayer(d_model, target_vocab_size)
        self.encoder_norm = LayerNorm(d_model)
        self.decoder_norm = LayerNorm(d_model)
        
    def encode(self, x, encoder_mask):
        x = self.source_token_embedding(x)
        x = self.source_psition_encoding(x)
        for encoder in self.encoder_module_list:
            x = encoder(x, encoder_mask)
        return self.encoder_norm(x)

    def decode(self, x, encoder_output, encoder_mask, decoder_mask):
        x = self.target_token_embedding(x)
        x = self.target_psition_encoding(x)
        for decoder in self.decoder_module_list:
            x = decoder(x, encoder_output, encoder_mask, decoder_mask)
        return self.decoder_norm(x)

    def project(self, x):
        x = self.projection(x)
        return x


def build_transformer(
    source_vocab_size: int,
    target_vocab_size: int,
    source_seq_len: int,
    target_seq_len: int,
    d_model: int,
    h: int,
    dropout: float,
    n_encoder: int,
    n_decoder: int,
):

    transformer = TransformerBlock(
        source_vocab_size=source_vocab_size,
        target_vocab_size=target_vocab_size,
        source_seq_len=source_seq_len,
        target_seq_len=target_seq_len,
        d_model=d_model,
        h=h,
        dropout=dropout,
        n_encoder=n_encoder,
        n_decoder=n_decoder,
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer