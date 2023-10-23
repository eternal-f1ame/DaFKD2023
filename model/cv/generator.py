import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, input_size, num_feature):
        super(Generator, self).__init__()
        self.fc = nn.Linear(input_size, num_feature)  # batch, 3136=1x56x56
        self.br = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(True)
        )
        self.downsample1 = nn.Sequential(
            nn.Conv2d(1, 50, 3, stride=1, padding=1),  # batch, 50, 56, 56
            nn.BatchNorm2d(50),
            nn.ReLU(True)
        )
        self.downsample2 = nn.Sequential(
            nn.Conv2d(50, 25, 3, stride=1, padding=1),  # batch, 25, 56, 56
            nn.BatchNorm2d(25),
            nn.ReLU(True)
        )
        self.downsample3 = nn.Sequential(
            nn.Conv2d(25, 1, 2, stride=2),  # batch, 1, 28, 28
            nn.Tanh()
        )

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 1, 56, 56)
        x = self.br(x)
        x = self.downsample1(x)
        x = self.downsample2(x)
        x = self.downsample3(x)
        return x

def weights_init(m):
    if type(m) in [nn.ConvTranspose2d, nn.Conv2d]:
        nn.init.xavier_normal_(m.weight)
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.constant_(m.bias, 0)
        
class GeneratorMNIST(nn.Module):
    def __init__(self, noise_dimension):
        super(GeneratorMNIST, self).__init__()
        self.noise_dimension = noise_dimension
        self.n_channel = 1
        self.n_g_feature = 64
        self.module = nn.Sequential(
            nn.ConvTranspose2d(noise_dimension, 4 * self.n_g_feature, kernel_size=4, bias=False),
            nn.BatchNorm2d(4 * self.n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(4 * self.n_g_feature, 2 * self.n_g_feature, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(2 * self.n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(2 * self.n_g_feature, self.n_g_feature, kernel_size=4, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(self.n_g_feature),
            nn.ReLU(),

            nn.ConvTranspose2d(self.n_g_feature, self.n_channel, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        x = self.module(x)
        return x
