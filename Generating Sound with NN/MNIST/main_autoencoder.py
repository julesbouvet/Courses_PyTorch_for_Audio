from autoencoder import AutoEncoder,VAE, generate_dataloader, train_model
from torch import save
from torchsummary import summary


# creating the autoencoder

#autoencoder = AutoEncoder()
autoencoder = VAE()

# visualising it
summary(autoencoder, input_size=(1, 28, 28))

# generating the train data
train_loader = generate_dataloader(batch_size=64, nb_train=200)

# training the model
trained_autoencoder = train_model(autoencoder, epochs=300, train_loader=train_loader, lr=0.01)

# saving the model
save(trained_autoencoder.state_dict(), 'my_autoencoder.pth')


