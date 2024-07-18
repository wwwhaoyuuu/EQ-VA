import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.out_dim = output_size
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1, 3)
        x = x.reshape((x.size(0), x.size(1), -1))

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = x.transpose(1, 2)
        x = F.avg_pool1d(x, x.size(2))
        x = x.squeeze(2)
        return x


def get_model_default_params():
    return dict(
            input_size=62 * 5, hidden_size1=64, hidden_size2=128, output_size=138
    )


@register_model
def MLP_classifier(pretrained=False, **kwargs):
    config = get_model_default_params()
    config["output_size"] = kwargs["num_classes"]
    print("MLP classifier parameters:", config)
    model = MLP(**config)

    return model


if __name__ == '__main__':
    x = torch.randn((2, 62, 1, 5)).float()
    model = MLP_classifier(num_classes=138)
    out = model(x)
    print(out.shape)
