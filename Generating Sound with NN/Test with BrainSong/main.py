from audio_autoencoder import VAE
from torchsummary import summary
from training import train_model

model = VAE()
summary(model, input_size=(2, 94080))
#model_trained = train_model(model, batchsize=4)

