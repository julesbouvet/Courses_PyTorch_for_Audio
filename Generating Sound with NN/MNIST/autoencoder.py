import torch.nn as nn
import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from torchvision.utils import save_image
from torch.nn import functional as F

        ########################################
        #                                      #
        #          VANILLA AUTOENCODER         #
        #                                      #
        ########################################

class AutoEncoder(nn.Module):

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1,32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Flatten(1, -1),
            nn.Linear(3136, 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(2, 3136),
            nn.Unflatten(1, (64, 7, 7)),

            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

        ########################################
        #                                      #
        #       VARIATIONAL AUTOENCODER        #
        #                                      #
        ########################################

class VAE(nn.Module):

    def __init__(self, h_dim=3136, z_dim=32):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Flatten(1, -1),
        )

        self.fc1 = nn.Linear(h_dim, z_dim)
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(z_dim, h_dim)

        self.decoder = nn.Sequential(
            #nn.Linear(2, 3136),
            nn.Unflatten(1, (64, 7, 7)),

            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.ConvTranspose2d(32, 1, 3, stride=1, padding=1),
            nn.Sigmoid()
        )
    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


    def forward(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        z = self.fc3(z)

        return self.decoder(z), mu, logvar

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        # return torch.normal(mu, std)
        esp = torch.randn(*mu.size())
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), self.fc2(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def representation(self, x):
        return self.bottleneck(self.encoder(x))[0]

        ########################################
        #                                      #
        #              FUNCTION                #
        #                                      #
        ########################################


def to_img(x):
    x = 0.5 * (x + 1)
    x = x.clamp(0, 1)
    x = x.view(x.size(0), 1, 28, 28)
    return x


def generate_dataloader(batch_size, nb_train):

    mnist = pd.read_csv("train.csv")
    y = mnist.iloc[:, 0].values
    X = mnist.iloc[:, 1:].values

    X, y = shuffle(X, y)

    X = np.array(X)[:nb_train, np.newaxis, :]
    y = y[:nb_train]

    X = np.reshape(X, (nb_train, 1, 28, 28))

    X = torch.Tensor(X)
    y = torch.Tensor(y)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size)

    return loader

def loss_fn(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    BCE = F.mse_loss(recon_x, x, reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD


def train_model(autoencoder, epochs, train_loader, lr):

    # optimizer
    optimizer = optim.Adam(autoencoder.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in train_loader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            batch_features = Variable(batch_features, requires_grad=True)
            outputs, mu, logvar = autoencoder(batch_features)


            # compute training reconstruction loss
            train_loss = loss_fn(outputs, batch_features, mu, logvar)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(train_loader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

        if epoch % (int(epochs/10)) == 0:
            pic = to_img(outputs.cpu().data)
            save_image(pic, 'dc_img/image_{}.png'.format(epoch))

    # visualiser le résultat final
    pic = to_img(outputs.cpu().data)
    save_image(pic, 'dc_img/image_{}.png'.format(epoch))

    print("Trained Model!")

    return autoencoder

