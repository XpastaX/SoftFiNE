import torch.nn as nn


class BaseClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, drop_out=0, name=None):
        super(BaseClassifier, self).__init__()
        self.name = name
        ModuleList = [nn.Dropout(p=drop_out)]
        if len(hidden_size) == 0:
            ModuleList.append(nn.Linear(input_size, output_size))
        else:
            for i in range(len(hidden_size)):
                hidden_size[i] = int(hidden_size[i])
            for i, h in enumerate(hidden_size):
                if i == 0:
                    ModuleList.append(nn.Linear(input_size, h))
                    ModuleList.append(nn.GELU())
                else:
                    ModuleList.append(nn.Linear(hidden_size[i - 1], h))
                    ModuleList.append(nn.GELU())
            ModuleList.append(nn.Linear(hidden_size[-1], output_size))

        self.MLP = nn.Sequential(*ModuleList)

    def forward(self, x):
        x = self.MLP(x)
        return x
