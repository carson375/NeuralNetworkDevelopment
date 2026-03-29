import torch
import torch.nn as nn

class ConstellationNet(nn.Module):
    """
    Multi-layer Perceptron for constellation classification from star data.
    """
    def __init__(self, nin=8, nout=1):
        super(ConstellationNet, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(0.20)

        # Encoder-decoder style symmetric architecture as found in original notebook
        self.layers = nn.Sequential(
            nn.Linear(nin, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            # Symmetric descent
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, nout),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.layers(x)

if __name__ == "__main__":
    # Test model with dummy input
    model = ConstellationNet()
    dummy_input = torch.randn(1, 8)
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
