import math
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.layers import trunc_normal_

from module.model_basic_modules import ResBlock

def patch_embedding(image, patch_size):
    embedding = rearrange(image, "b c d h w -> b (d h w) c", h=patch_size)
    return embedding

def patch_recover(embedding, patch_size):
    image = rearrange(embedding, "b (d h w) c -> b c d h w", d=patch_size, h=patch_size)
    return image

class Multi_Head_Attention(nn.Module):
    def __init__(self, embed_size, attention_heads, dropout=0.3):
        super().__init__()
        self.attention_heads = attention_heads
        self.head_size = embed_size // attention_heads
        self.qkv = nn.Linear(embed_size, embed_size*3)
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # x(B, N*N*N, C)
        B, N, C = x.shape
        qkv = self.qkv(x)
        qkv = qkv.contiguous().view(B, N, self.attention_heads, self.head_size, 3).permute(4, 0, 2, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-1, -2))
        attn = attn / math.sqrt(self.head_size)
        attn = self.dropout(self.softmax(attn))
        attn = torch.matmul(attn, v).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B, N, -1)
        attn = self.dropout(self.out(attn))
        return attn, q, k, v


class MLP(nn.Module):
    def __init__(self, embed_size, hidden_size, dropout=0.3):
        super().__init__()
        self.linear1 = nn.Linear(embed_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, embed_size)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.linear1(x)))
        x = self.dropout(self.linear2(x))
        return x

class Multi_Attention_Block(nn.Module):
    def __init__(self, embed_size, attention_heads, dropout=0.3):
        super().__init__()
        self.layernorm = nn.LayerNorm(embed_size)
        self.attention = Multi_Head_Attention(embed_size=embed_size, attention_heads=attention_heads, dropout=dropout)

    def forward(self, x):
        x1 = x
        x, q, k, v = self.attention(self.layernorm(x))
        x = x + x1

        return x, q, k, v

class Multi_Attention_Transformer(nn.Module):
    def __init__(self, layers, embed_size, attention_heads, dropout=0.3):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(layers):
            self.layers.append(Multi_Attention_Block(embed_size=embed_size, attention_heads=attention_heads, dropout=dropout))
        self.layernorm = nn.LayerNorm(embed_size)

    def forward(self, x):
        for layer in self.layers:
            x, q, k, v = layer(x)
        return self.layernorm(x), q, k, v


class MFC_model(nn.Module):   # x (B, C, D, H, W)
    def __init__(self, mfc_embed_size, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.compression_size = mfc_embed_size // 2
        self.channel_avg = nn.AdaptiveAvgPool3d(1)
        self.position_embedding = nn.Parameter(torch.zeros(1, mfc_embed_size, patch_size * patch_size * patch_size))
        trunc_normal_(self.position_embedding, std=.02)
        self.layer_norm = nn.LayerNorm(mfc_embed_size)
        self.linear = nn.Linear(mfc_embed_size, self.compression_size)
        self.restnet = ResBlock(in_channels=mfc_embed_size, out_channels=self.compression_size)
        self.norm = nn.LayerNorm(self.compression_size)

    def forward(self, x):
        x_conv = self.restnet(x)
        channel_avg = self.channel_avg(x)
        channel_avg = channel_avg.squeeze(-1).squeeze(-1).transpose(-1, -2)
        x = patch_embedding(x, patch_size=self.patch_size)
        x_conv = patch_embedding(x_conv, patch_size=self.patch_size)
        B, N, C = x.shape
        channel_avg = channel_avg.expand(-1, N, -1)
        x_linear = x + channel_avg + self.position_embedding.transpose(-1, -2)
        x_linear = self.linear(self.layer_norm(x_linear))
        x = x_conv + x_linear
        return self.norm(x)


class MFI_Block(nn.Module):
    def __init__(self, mfi_embed_size, mfi_attention_heads, mfi_hidden_size, dropout=0.3):
        super().__init__()
        self.layernorm = nn.LayerNorm(mfi_embed_size)
        self.mfi_qk = nn.Linear(mfi_embed_size, mfi_embed_size * 2)
        self.mfi_v = nn.Linear(mfi_embed_size, mfi_embed_size)
        self.attention_heads = mfi_attention_heads
        self.head_size = mfi_embed_size // mfi_attention_heads
        self.softmax = nn.Softmax(dim=-1)
        self.out = nn.Linear(mfi_embed_size, mfi_embed_size)
        self.attn_dropout = nn.Dropout(dropout)
        self.a = 0.7
        self.b = 0.3

        self.mfi_mlp = MLP(embed_size=mfi_embed_size, hidden_size=mfi_hidden_size)

    def forward(self, Query, Key, Value, x):   # x (B, N, C)(1 64 512)
        B, N, C = x.shape
        x1 = x
        x = self.layernorm(x)
        mfi_qk = self.mfi_qk(x).contiguous().view(B, N, self.attention_heads, self.head_size, 2).permute(4, 0, 2, 1, 3)
        Q, K = mfi_qk[0], mfi_qk[1]
        Query = self.a * Q + self.b * Query
        Key = self.a * K + self.b * Key
        Value = (self.b * Value + self.a *
                 self.mfi_v(x).contiguous().view(B, N, self.attention_heads, self.head_size).permute(0, 2, 1, 3))

        attn = torch.matmul(Query, Key.transpose(-1, -2))
        attn = attn / math.sqrt(self.head_size)
        attn = self.attn_dropout(self.softmax(attn))
        attn = torch.matmul(attn, Value).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B, N, -1)
        attn = self.attn_dropout(self.out(attn))
        x = x1 + attn

        x2 = x
        x = self.layernorm(x)
        x = self.mfi_mlp(x)
        return x + x2, Query, Key, Value

class MFI_model(nn.Module):
    def __init__(self, mfi_layers, mfi_embed_size, mfi_attention_heads, mfi_hidden_size, dropout=0.3):
        super().__init__()
        self.attention_heads = mfi_attention_heads
        self.head_size = mfi_embed_size // mfi_attention_heads
        self.layers = nn.ModuleList()
        for i in range(mfi_layers):
            self.layers.append(MFI_Block(mfi_embed_size=mfi_embed_size, mfi_attention_heads=mfi_attention_heads,
                                         mfi_hidden_size=mfi_hidden_size, dropout=dropout))
        self.layernorm = nn.LayerNorm(mfi_embed_size)

    def forward(self, Query, Key, Value, x):
        B, N, C = x.shape
        Query, Key, Value = (Query.contiguous().view(B, N, self.attention_heads, self.head_size).permute(0, 2, 1, 3),
                             Key.contiguous().view(B, N, self.attention_heads, self.head_size).permute(0, 2, 1, 3),
                             Value.contiguous().view(B, N, self.attention_heads, self.head_size).permute(0, 2, 1, 3))
        for layer in self.layers:
            x, Query, Key, Value = layer(Query, Key, Value, x)
        x = self.layernorm(x)
        return x


class MFCI_model(nn.Module):
    def __init__(self, mfc_embed_size, patch_size, mode_layers, mode_embed_size, mode_attention_heads,
                 mfi_layers, mfi_embed_size, mfi_attention_heads, mfi_hidden_size, dropout=0.3):
        super().__init__()
        patch_size = patch_size[0]
        self.patch_size = patch_size
        self.mfc = MFC_model(mfc_embed_size=mfc_embed_size, patch_size=patch_size)
        self.t1_transformer = Multi_Attention_Transformer(layers=mode_layers, embed_size=mode_embed_size,
                                                          attention_heads=mode_attention_heads)
        self.t1ce_transformer = Multi_Attention_Transformer(layers=mode_layers, embed_size=mode_embed_size,
                                                            attention_heads=mode_attention_heads)
        self.t2_transformer = Multi_Attention_Transformer(layers=mode_layers, embed_size=mode_embed_size,
                                                          attention_heads=mode_attention_heads)
        self.flair_transformer = Multi_Attention_Transformer(layers=mode_layers, embed_size=mode_embed_size,
                                                             attention_heads=mode_attention_heads)
        self.q_avg = nn.AdaptiveAvgPool1d(1)
        self.k_avg = nn.AdaptiveAvgPool1d(1)
        self.linear_q = nn.Sequential(
            nn.LayerNorm(mode_embed_size * 4), nn.Linear(mode_embed_size * 4, mode_embed_size * 2),
            MLP(embed_size=mode_embed_size*2, hidden_size=mode_embed_size*4, dropout=dropout)
        )
        self.linear_k = nn.Sequential(
            nn.LayerNorm(mode_embed_size * 4), nn.Linear(mode_embed_size * 4, mode_embed_size * 2),
            MLP(embed_size=mode_embed_size*2, hidden_size=mode_embed_size*4, dropout=dropout)
        )
        self.linear_v = nn.Sequential(
            nn.LayerNorm(mode_embed_size), nn.Linear(mode_embed_size, mode_embed_size * 2),
            MLP(embed_size=mode_embed_size*2, hidden_size=mode_embed_size*4, dropout=dropout)
        )
        self.mfi = MFI_model(mfi_layers=mfi_layers, mfi_embed_size=mfi_embed_size, mfi_attention_heads=mfi_attention_heads,
                             mfi_hidden_size=mfi_hidden_size, dropout=dropout)


    def forward(self, t1, t1ce, t2, flair):  # x (B, C, D, H, W)
        F = torch.cat([t1, t1ce, t2, flair], dim=1)
        F = self.mfc(F)   # (B, N, C)(1, 192, 4096)
        # print(F.shape)

        t1, t1ce, t2, flair = (patch_embedding(t1, patch_size=self.patch_size), patch_embedding(t1ce, patch_size=self.patch_size),
                               patch_embedding(t2, patch_size=self.patch_size), patch_embedding(flair, patch_size=self.patch_size))
        _, t1_q, t1_k, t1_v = self.t1_transformer(t1)
        _, t1ce_q, t1ce_k, t1ce_v = self.t1ce_transformer(t1ce)
        _, t2_q, t2_k, t2_v = self.t2_transformer(t2)
        _, flair_q, flair_k, flair_v = self.flair_transformer(flair)

        B, _, N, _ = t1_q.shape
        F_Q = torch.cat([t1_q.permute(0, 2, 1, 3).contiguous().view(B, N, -1),
                         t1ce_q.permute(0, 2, 1, 3).contiguous().view(B, N, -1),
                         t2_q.permute(0, 2, 1, 3).contiguous().view(B, N, -1),
                         flair_q.permute(0, 2, 1, 3).contiguous().view(B, N, -1)], dim=2)
        F_K = torch.cat([t1_k.permute(0, 2, 1, 3).contiguous().view(B, N, -1),
                         t1ce_k.permute(0, 2, 1, 3).contiguous().view(B, N, -1),
                         t2_k.permute(0, 2, 1, 3).contiguous().view(B, N, -1),
                         flair_k.permute(0, 2, 1, 3).contiguous().view(B, N, -1)], dim=2)
        F_V = (t1_v.permute(0, 2, 1, 3).contiguous().view(B, N, -1) + t1ce_v.permute(0, 2, 1, 3).contiguous().view(B, N, -1)
               + t2_v.permute(0, 2, 1, 3).contiguous().view(B, N, -1) + flair_v.permute(0, 2, 1, 3).contiguous().view(B, N, -1))
        F_Q_avg = self.q_avg(F_Q.transpose(-1, -2)).transpose(-1, -2)
        F_K_avg = self.k_avg(F_K.transpose(-1, -2)).transpose(-1, -2)
        F_Q = self.linear_q(F_Q + F_Q_avg)
        F_K = self.linear_k(F_K + F_K_avg)
        F_V = self.linear_v(F_V)

        x = self.mfi(F_Q, F_K, F_V, F)
        return patch_recover(x, patch_size=self.patch_size)    # (B, C, D, H, W)