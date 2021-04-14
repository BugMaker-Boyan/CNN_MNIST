import torch.nn as nn
import torch
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=16,
                      kernel_size=5,
                      stride=1,
                      padding=2),   # 16 28 28
            nn.BatchNorm2d(16),
            nn.ReLU()
        )
        self.pool1 = nn.MaxPool2d(2)    # 16 14 14
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=16,
                      out_channels=32,
                      kernel_size=3,
                      stride=1,
                      padding=0),    # 32 12 12
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=0),   # 64 10 10
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool2d(2)  # 64 5 5
        self.fc = nn.Linear(64*5*5, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.pool1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.pool2(out)
        # print(out.size())
        out = out.view([out.size()[0], -1])
        out = self.fc(out)
        return F.log_softmax(out, dim=-1)

# cnn = CNN()
# print(cnn(torch.rand(1, 1, 28, 28)))
# print(cnn)
