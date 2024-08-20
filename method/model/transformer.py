import torch
from timm.models import register_model
from torch import nn
from einops import rearrange, repeat


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                    PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                    PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class TF(nn.Module):
    def __init__(self, *, num_classes, feature_dim, embed_dim, depth, heads, mlp_dim, dim_head=64,
                 dropout=0., emb_dropout=0.):
        super().__init__()

        self.eeg_proj = nn.Linear(feature_dim, embed_dim)

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, 128 + 1, embed_dim))

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(embed_dim, depth, heads, dim_head, mlp_dim, dropout)
        self.norm_layer = nn.LayerNorm(embed_dim)

        self.lm_head = nn.Linear(embed_dim, num_classes)

    def forward(self, x, input_chans=None):
        bs, n_channels, n_seconds, feature_dim = x.shape
        x = x.flatten(1, 2)
        x = self.eeg_proj(x)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=bs)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n_channels * n_seconds + 1)]

        x = self.dropout(x)
        x = self.transformer(x)
        x = self.norm_layer(x)
        x = x[:, 0]
        return self.lm_head(x)


def get_model_default_params():
    return dict(
            feature_dim=5, embed_dim=256, depth=12, heads=10, mlp_dim=2048, dim_head=64, dropout=0.01, emb_dropout=0.01
    )


@register_model
def TF_classifier(pretrained=False, **kwargs):
    config = get_model_default_params()
    config["num_classes"] = kwargs["num_classes"]
    config['feature_dim'] = kwargs['feature_dim']
    config["depth"] = 12
    config["heads"] = 10
    print("TF classifier parameters:", config)
    model = TF(**config)

    return model


if __name__ == '__main__':
    model = TF_classifier(num_classes=138, feature_dim=5)
    x = torch.randn(2, 62, 1, 5)
    y = model(x)
    print(y.shape)
