import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import trunc_normal_
from timm.models.registry import register_model
from utils.norm_ema_quantizer import NormEMAVectorQuantizer
from backbone_vqdep import DET
from modeling_VAT import VAT_classifier
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from collections import Counter
import numpy as np
from matplotlib.colors import ListedColormap
import os


class VQDEP(nn.Module):
    def __init__(self, encoder_config, decoder_config, n_embed=8192, embed_dim=32, decay=0.99,
                 quantize_kmeans_init=True, decoder_out_dim=256, smooth_l1_loss=False, **kwargs):
        super().__init__()
        print(kwargs)

        # encoder & decode params
        print('Final Encoder config', encoder_config)
        self.encoder = DET(**encoder_config)
        print('Final Decoder config', decoder_config)
        self.decoder = DET(**decoder_config)

        self.quantize = NormEMAVectorQuantizer(
            n_embed=n_embed, embedding_dim=embed_dim, beta=1.0, kmeans_init=quantize_kmeans_init, decay=decay,
        )

        self.decoder_out_dim = decoder_out_dim

        # task layer
        self.encode_task_layer = nn.Sequential(
            nn.Linear(encoder_config['embed_dim'], encoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(encoder_config['embed_dim'], embed_dim)  # for quantize
        )
        self.decode_task_layer = nn.Sequential(
            nn.Linear(decoder_config['embed_dim'], decoder_config['embed_dim']),
            nn.Tanh(),
            nn.Linear(decoder_config['embed_dim'], self.decoder_out_dim),
        )
        self.kwargs = kwargs

        self.encode_task_layer.apply(self._init_weights)
        self.decode_task_layer.apply(self._init_weights)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'quantize.embedding.weight', 'decoder.cls_token', 'decoder.pos_embed', 'decoder.time_embed',
                'encoder.cls_token', 'encoder.pos_embed', 'encoder.time_embed'}

    @property
    def device(self):
        return self.decoder.cls_token.device

    def get_number_of_tokens(self):
        return self.quantize.n_e

    def get_tokens(self, data, input_chans=None, **kwargs):
        quantize, embed_ind, loss = self.encode(data, input_chans=input_chans)
        output = {}
        # embed_ind shape[b,c * t]
        output['token'] = embed_ind.view(data.shape[0], -1)
        output['EEG'] = data
        # output['quantize'] = rearrange(quantize, 'b c t d -> b (c t) d')
        output['quantize'] = quantize

        return output

    def encode(self, x, input_chans=None):
        bs, n_channels, n_second, feature_dim = x.shape
        encoder_features = self.encoder(x, input_chans, return_patch_tokens=True)

        with torch.cuda.amp.autocast(enabled=False):
            to_quantizer_features = self.encode_task_layer(encoder_features.type_as(self.encode_task_layer[-1].weight))

        # reshape for quantizer
        to_quantizer_features = rearrange(to_quantizer_features, 'b (c t) d -> b d c t', c=n_channels, t=n_second)
        # quantize shape[b,c,t,d],embed_ind shape[N=b*c*t]
        quantize, loss, embed_ind = self.quantize(to_quantizer_features)

        return quantize, embed_ind, loss

    def decode(self, quantize, input_chans=None, **kwargs):
        decoder_features = self.decoder(quantize, input_chans, return_patch_tokens=True)
        rec = self.decode_task_layer(decoder_features)
        return rec

    def get_codebook_indices(self, x, input_chans=None, **kwargs):
        # for LaBraM pre-training
        return self.get_tokens(x, input_chans, **kwargs)['token']

    def get_codebook_quantize(self, x, input_chans=None, **kwargs):
        # quantize shape[b,c,t,d]
        return self.get_tokens(x, input_chans, **kwargs)['quantize']

    def visualize_tsne(self, data, codes, labels, epoch, step=1, save_path=None):
        os.makedirs(save_path, exist_ok=True)

        tsne = TSNE(n_components=2, perplexity=40, n_iter=1000)
        codes_2d = tsne.fit_transform(codes)
        
        plt.figure(figsize=(20, 20))

        # 找出标签中出现次数最多的 20 个标签
        label_counter = Counter(labels)
        top_20_labels = [label for label, _ in label_counter.most_common(20)]
        
        # 过滤出这 20 个标签的数据
        mask = np.isin(labels, top_20_labels)
        filtered_codes_2d = codes_2d[mask]
        filtered_labels = labels[mask]

        # 把图片拿出来
        filtered_data = data[mask]
        
        # 使用一个连续的 colormap 并生成足够多的颜色
        num_labels = len(top_20_labels)
        base_cmap = plt.get_cmap('tab20', num_labels)
        colors = base_cmap(np.linspace(0, 1, num_labels))
        cmap = ListedColormap(colors)
        
        # 创建颜色数组，确保标签在有效范围内
        label_to_color = {label: colors[i] for i, label in enumerate(top_20_labels)}
        color_array = np.array([label_to_color[label] for label in filtered_labels])
        
        # 绘制散点图，并显示图例
        scatter = plt.scatter(filtered_codes_2d[:, 0], filtered_codes_2d[:, 1], c=color_array, marker='o', alpha=0.8, s=100)
        handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=15, label=str(label)) for label, color in label_to_color.items()]
        
        # 设置图例并将其放置在图外
        plt.legend(handles=handles, title='Code idx', bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize=10, title_fontsize=12)
        plt.title(f"t-SNE visualization of Codes", fontsize=24)
        plt.subplots_adjust(right=0.8)  # 调整图例的显示位置

        # 设置坐标轴的缩放比例
        plt.xlim(np.min(filtered_codes_2d[:, 0]) * 1.1, np.max(filtered_codes_2d[:, 0]) * 1.1)
        plt.ylim(np.min(filtered_codes_2d[:, 1]) * 1.1, np.max(filtered_codes_2d[:, 1]) * 1.1)

        plt.savefig(os.path.join(save_path, f"tsne_epoch_{epoch}_step{step}.png"))
        plt.close()

        # # 保存对应的原始数据图像，每个标签最多保存三张
        # label_save_count = Counter()
        # for i, (datum, label) in enumerate(zip(filtered_data, filtered_labels)):
        #     if label_save_count[label] >= 3:
        #         continue  # 跳过超过三张的标签
        #     plt.figure()
        #     plt.plot(datum, color='red')
        #     plt.title(f"Label {label}")
        #     plt.savefig(os.path.join(save_path, f"label_{label}_index_{i}.png"))
        #     plt.close()
        #     label_save_count[label] += 1  # 更新计数器
            

    def calculate_rec_loss(self, rec, target):
        target = rearrange(target, 'b c t d -> b (c t) d')
        rec_loss = self.loss_fn(rec, target)
        return rec_loss

    def std_norm(self, x):
        mean = torch.mean(x, dim=(1, 2, 3), keepdim=True)
        std = torch.std(x, dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / std
        return x

    def forward(self, x, input_chans=None, **kwargs):
        """
            EEG input shape: [B,n_channels,n_second,feature_dim]
        """
        quantize, embed_ind, emb_loss = self.encode(x, input_chans)
        xrec = self.decode(quantize, input_chans)

        rec_loss = self.calculate_rec_loss(xrec, x)
        loss = emb_loss + rec_loss

        log = {}
        split = "train" if self.training else "val"
        log[f'{split}/quant_loss'] = emb_loss.detach().mean()
        log[f'{split}/rec_loss'] = rec_loss.detach().mean()
        log[f'{split}/total_loss'] = loss.detach().mean()

        return loss, log, xrec


def get_model_default_params():
    return dict(
        feature_dim=5, embed_dim=512, depth=12, heads=10, mlp_dim=256, dim_head=64, dropout=0.01, emb_dropout=0.01
    )


@register_model
def vqdep_vocab_1k_dim_32(pretrained=False, pretrained_weight=None, as_tokenzer=False,
                          n_code=1024, code_dim=32, **kwargs):
    encoder_config, decoder_config = get_model_default_params(), get_model_default_params()

    # encoder settings
    encoder_config['feature_dim'] = 5
    encoder_config['depth'] = 12
    encoder_config['heads'] = 10

    # decoder settings
    decoder_config['feature_dim'] = code_dim
    decoder_config['depth'] = 3
    decoder_config['heads'] = 10
    decoder_out_dim = 5

    model = VQDEP(encoder_config, decoder_config, n_code, code_dim, decoder_out_dim=decoder_out_dim, **kwargs)

    if as_tokenzer:
        assert pretrained
        assert pretrained_weight is not None

        if pretrained_weight.startswith('https'):
            weights = torch.hub.load_state_dict_from_url(pretrained_weight, map_location='cpu', check_hash=True)
        else:
            weights = torch.load(pretrained_weight, map_location='cpu')

        if 'model' in weights:
            weights = weights['model']
        else:
            weights = weights["state_dict"]
        keys = list(weights.keys())

        for k in keys:
            if k.startswith("loss") or k.startswith("teacher") or k.startswith("scaling"):
                del weights[k]
        model.load_state_dict(weights)
    return model


# 设置少一点的code，这样能强迫模型学习泛化特征
if __name__ == '__main__':
    model1 = vqdep_vocab_1k_dim_32(pretrained=False, pretrained_weight=None, as_tokenzer=False)
    input = torch.randn((2, 62, 1, 5))
    in_chans = torch.arange(0, 63)
    output = model1(input, in_chans)
    # print(output)
    quantize = model1.get_codebook_quantize(input, in_chans)
    print(quantize.shape)

    model2 = VAT_classifier(num_classes=138)
    output = model2(quantize, in_chans)
    print(output.shape)
    targets = torch.randn((2,138))
    loss = torch.nn.CrossEntropyLoss()(output, targets)
    print(loss)
