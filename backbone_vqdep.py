import torch
from torch import nn
from einops import rearrange, repeat


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class RoPE(nn.Module):
    def __init__(self):
        super(RoPE, self).__init__()

    def sinusoidal_position_embedding(
        self, batch_size, num_heads, max_len, output_dim, device
    ):
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, output_dim, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / output_dim)
        )
        embeddings = torch.zeros(max_len, output_dim, device=device)
        embeddings[:, 0::2] = torch.sin(position * div_term)
        embeddings[:, 1::2] = torch.cos(position * div_term)
        # shape: (1, 1, max_len, output_dim)
        embeddings = embeddings.unsqueeze(0).unsqueeze(0)
        # shape: (batch_size, num_heads, max_len, output_dim)
        embeddings = embeddings.repeat(batch_size, num_heads, 1, 1)
        return embeddings

    def forward(self, q, k):
        batch_size, num_heads, max_len, output_dim = q.shape
        device = q.device
        pos_emb = self.sinusoidal_position_embedding(
            batch_size, num_heads, max_len, output_dim, device
        )

        cos_pos = pos_emb[..., 1::2].repeat_interleave(2, dim=-1)
        sin_pos = pos_emb[..., ::2].repeat_interleave(2, dim=-1)

        q2 = torch.stack([-q[..., 1::2], q[..., ::2]], dim=-1).reshape(q.shape)
        k2 = torch.stack([-k[..., 1::2], k[..., ::2]], dim=-1).reshape(k.shape)

        q = q * cos_pos + q2 * sin_pos
        k = k * cos_pos + k2 * sin_pos

        return q, k


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        rope = RoPE()
        q, k = rope(q, k)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class DET(nn.Module):
    def __init__(
        self,
        *,
        feature_dim,
        embed_dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0
    ):
        super().__init__()
        self.eeg_proj = TokenEmbedding(c_in=feature_dim, d_model=embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.spa_embed = nn.Parameter(
            torch.randn(1, 128 + 1, embed_dim), requires_grad=True
        )
        self.tem_embed = nn.Parameter(torch.randn(1, 16, embed_dim), requires_grad=True)

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            embed_dim, depth, heads, dim_head, mlp_dim, dropout
        )

        self.norm_layer = nn.LayerNorm(embed_dim)

    def forward_feature(self, x, input_chans=None):
        bs, n_channels, n_seconds, feature_dim = x.shape
        x = x.flatten(1, 2)
        x = self.eeg_proj(x)

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=bs)
        x = torch.cat((cls_tokens, x), dim=1)

        if input_chans is not None:
            spa_embed_used = self.spa_embed[:, input_chans]
            spa_embed = (
                spa_embed_used[:, 1:, :]
                .unsqueeze(2)
                .expand(bs, -1, n_seconds, -1)
                .flatten(1, 2)
            )
            spa_embed = torch.cat(
                (spa_embed_used[:, 0:1, :].expand(bs, -1, -1), spa_embed), dim=1
            )
            x = x + spa_embed

        tem_embed = (
            self.tem_embed[:, 0:n_seconds, :]
            .unsqueeze(1)
            .expand(bs, n_channels, -1, -1)
            .flatten(1, 2)
        )
        x[:, 1:, :] += tem_embed

        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm_layer(x)
        return x

    def forward(
        self, x, input_chans=None, return_all_tokens=False, return_patch_tokens=False
    ):
        x = self.forward_feature(x, input_chans)
        if return_all_tokens:
            return x
        elif return_patch_tokens:
            return x[:, 1:]
        else:
            return x[:, 0]


if __name__ == "__main__":
    model = DET(feature_dim=5, embed_dim=200, depth=1, heads=4, mlp_dim=1024)
    x = torch.randn(2, 62, 5, 5)
    in_chans = torch.arange(0, 63)
    y = model(x, in_chans)
    print(y.shape)
