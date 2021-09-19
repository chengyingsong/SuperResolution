from torch import nn


class SRCNN(nn.Module):

    def __init__(self):
        super(SRCNN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=9, stride=1, padding=9 // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1, padding=0 // 2),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=3, kernel_size=5, stride=1, padding=5 // 2),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)
