import torch
import torch.nn as nn
from timm.models import register_model


class EEGNet(nn.Module):
    def __init__(self, num_classes, n_channels=62, feature_dim=5, dropout_rate=0.5, kern_length=16, F1=8, D=2, F2=16):
        super(EEGNet, self).__init__()

        # Layer 1: Temporal Convolution
        self.conv1 = nn.Conv2d(1, F1, (1, kern_length), padding=(0, kern_length // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, False)

        # Layer 2: Depthwise Convolution
        self.depthwise = nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, False)
        self.activ = nn.ELU()
        self.pooling1 = nn.AvgPool2d((1, 2))
        self.dropout1 = nn.Dropout(p=dropout_rate)

        # Layer 3: Separable Convolution
        self.separable_conv = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 8), padding=(0, 4), groups=F1 * D, bias=False),
                nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
                nn.BatchNorm2d(F2, False),
                nn.ELU(),
                nn.AvgPool2d((1, 2)),
                nn.Dropout(p=dropout_rate)
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(F2 * (feature_dim // 2), num_classes)

    def forward(self, x):
        # Ensure input is (batch_size, 1, n_channels, feature_dim)
        x = x.permute(0, 2, 1,
                      3)  # (batch_size, n_channels, 1, feature_dim) -> (batch_size, 1, n_channels, feature_dim)

        # Layer 1: Temporal Convolution
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.activ(x)

        # Layer 2: Depthwise Convolution
        x = self.depthwise(x)
        x = self.batchnorm2(x)
        x = self.activ(x)
        x = self.pooling1(x)
        x = self.dropout1(x)

        # Layer 3: Separable Convolution
        x = self.separable_conv(x)

        # Flatten and Fully Connected Layer
        x = self.flatten(x)
        x = self.fc(x)

        return x


def get_model_default_params():
    return dict(
            num_classes=138, n_channels=62, feature_dim=5, dropout_rate=0.5, kern_length=16, F1=8, D=2, F2=16
    )


@register_model
def EEGNet_classifier(pretrained=False, **kwargs):
    config = get_model_default_params()
    config["num_classes"] = kwargs["num_classes"]
    print("EEGNet classifier parameters:", config)
    model = EEGNet(**config)

    return model


if __name__ == "__main__":
    model = EEGNet_classifier(num_classes=138)
    x = torch.randn(8, 30, 1, 5)
    output = model(x)
    print(output.shape)
