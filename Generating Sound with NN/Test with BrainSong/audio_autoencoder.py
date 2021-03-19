import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()
        power = 4

        # Encoder

        self.conv1 = nn.Conv1d(2, 2 ** power, kernel_size=64, stride=4)
        self.bn1 = nn.BatchNorm1d(2 ** power)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(2 ** power, 2 ** power, 4)
        self.bn2 = nn.BatchNorm1d(2 ** power)
        self.pool2 = nn.MaxPool1d(4)

        self.conv3 = nn.Conv1d(2 ** power, 2 ** (power - 1), 4)
        self.bn3 = nn.BatchNorm1d(2 ** (power - 1))
        self.pool3 = nn.MaxPool1d(4)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(8*366, 4)

        # Decoder

        self.d_fc1 = nn.Linear(4, 8*366)

        self.unflatten = nn.Unflatten(1, (8, 366))
        self.d_conv2 = nn.ConvTranspose1d(2 ** (power-1), 2 ** power, 5)
        self.d_bn2 = nn.BatchNorm1d(2 ** power)
        self.d_pool2 = nn.MaxPool1d(4)

        self.d_conv3 = nn.ConvTranspose1d(2 ** power, 2 ** power, 5)
        self.d_bn3 = nn.BatchNorm1d(2 ** power)
        self.d_pool3 = nn.MaxPool1d(4)

        self.d_conv4 = nn.ConvTranspose1d(2 ** power, 2, 64, 5)

    def encoder(self, x):
        print('input', x.shape)
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)

        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        print(x.shape)

        # x = F.avg_pool1d(x, x.data.size()[2])
        # x = x.view(x.data.size()[0], x.data.size()[2], x.data.size()[1])
        x = self.flatten(x)
        x = self.fc1(x)
        print('sortie encoder', x.shape)
        # x = x.view(x.data.size()[0], x.data.size()[2])
        return x.data

    def decoder(self, x):
        print('decode',x.shape)
        x = self.d_fc1(x)
        x = self.unflatten(x)
        x = self.d_conv2(x)
        x = F.relu(self.d_bn2(x))
        x = self.d_pool2(x)

        x = self.d_conv3(x)
        x = F.relu(self.d_bn3(x))
        x = self.d_pool3(x)

        x = self.d_conv4(x)
        print('end', x.shape)

        return x

    def forward(self, x):
        print(x.shape)
        x = self.encoder(x)
        x = self.decoder(x)
        return x

