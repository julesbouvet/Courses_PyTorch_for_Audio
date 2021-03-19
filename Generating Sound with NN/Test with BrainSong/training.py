from audio_dataset import AudioDataSet
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.autograd import Variable


def train_model(model, batchsize, lr, epochs, npzfile='all_sample.npz'):

    dataset = AudioDataSet(npzfile)
    dataloader = DataLoader(dataset, batch_size=batchsize,shuffle=True)

    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        loss = 0
        for batch_features, _ in dataloader:
            # reshape mini-batch data to [N, 784] matrix
            # load it to the active device

            # reset the gradients back to zero
            # PyTorch accumulates gradients on subsequent backward passes
            optimizer.zero_grad()

            # compute reconstructions
            batch_features = Variable(batch_features, requires_grad=True)
            outputs, mu, logvar = model(batch_features)


            # compute training reconstruction loss
            train_loss = loss_fn(outputs, batch_features, mu, logvar)

            # compute accumulated gradients
            train_loss.backward()

            # perform parameter update based on current gradients
            optimizer.step()

            # add the mini-batch training loss to epoch loss
            loss += train_loss.item()

        # compute the epoch training loss
        loss = loss / len(dataloader)

        # display the epoch training loss
        print("epoch : {}/{}, loss = {:.6f}".format(epoch + 1, epochs, loss))

    print("Trained Model!")
    return model


