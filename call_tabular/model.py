import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_layer, drop_p,) -> None:
        super().__init__()
        modules = []

        
        input_layer = nn.Sequential(
             nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
        )
        modules.append(input_layer)

        for i in range(n_layer-2):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(drop_p),
                )
            )

        output_layer = nn.Sequential(
            nn.Linear(hidden_size, output_size),
            nn.Sigmoid()
        )
        modules.append(output_layer)

        self.model = nn.ModuleList(modules)

    def forward(self, x):
        for layer in self.model:
            x = layer(x)
        return x