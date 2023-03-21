import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, n_layer, drop_p,) -> None:
        super().__init__()
        input_layer = nn.Sequential(
             nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
        )

        output_layer = nn.Linear(hidden_size, output_size),
        

        hidden_layers = []
        for i in range(n_layer-2):
            hidden_layers.append(
                nn.Sequential(
                    nn.Linear(hidden_size, hidden_size),
                    nn.ReLU(),
                    nn.Dropout(drop_p),
                )
            )

        modules = [input_layer] + hidden_layers + [output_layer]
        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)