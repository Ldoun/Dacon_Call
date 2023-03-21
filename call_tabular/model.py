import torch.nn as nn

class DNN(nn.Module):
    def __init__(self, input_size, hidden_size,output_size, drop_p, ) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(drop_p),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, x):
        return self.model(x)