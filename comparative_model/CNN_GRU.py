import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model


class CNN_GRU(nn.Module):
    def __init__(self, num_channels, time, num_classes):
        super(CNN_GRU, self).__init__()
        self.conv1 = nn.Conv2d(time, 8, kernel_size=(3, 3), padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), padding=1)
        self.gru = nn.GRU(5 * num_channels, 256, num_layers=3, batch_first=True, bidirectional=False)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))

        x = x.reshape(x.size(0), x.size(1), -1)
        x, _ = self.gru(x)
        x = torch.transpose(x, 1, 2)
        x = F.max_pool1d(x, x.size(2))
        x = x.squeeze(2)
        x = self.fc(x)

        return x


def get_model_default_params():
    return dict(
            num_classes=138, time=1, num_channels=62
    )


@register_model
def CNN_GRU_classifier(pretrained=False, **kwargs):
    config = get_model_default_params()
    config["num_classes"] = kwargs["num_classes"]
    print("CNN-GRU classifier parameters:", config)
    model = CNN_GRU(**config)

    return model


if __name__ == '__main__':
    x = torch.randn((2, 30, 1, 5)).float()
    model = CNN_GRU_classifier(num_classes=138)
    out = model(x)
    print(out.shape)