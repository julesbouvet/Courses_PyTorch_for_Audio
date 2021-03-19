import json
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

DATA_PATH = "data.json"


def load_data(data_path):
    """Loads training dataset from json file.
        :param data_path (str): Path to json file containing data
        :return X (ndarray): Inputs
        :return y (ndarray): Targets
    """

    with open(data_path, "r") as fp:
        data = json.load(fp)

    X = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return X, y


def plot_history(history):
    """Plots accuracy/loss for training/validation set as a function of the epochs
        :param history: Training history of model
        :return:
    """

    fig, axs = plt.subplots(2)

    # create accuracy sublpot
    axs[0].plot(history.history["accuracy"], label="train accuracy")
    axs[0].plot(history.history["val_accuracy"], label="test accuracy")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend(loc="lower right")
    axs[0].set_title("Accuracy eval")

    # create error sublpot
    axs[1].plot(history.history["loss"], label="train error")
    axs[1].plot(history.history["val_loss"], label="test error")
    axs[1].set_ylabel("Error")
    axs[1].set_xlabel("Epoch")
    axs[1].legend(loc="upper right")
    axs[1].set_title("Error eval")

    plt.show()


def prepare_datasets(test_size, validation_size):
    """Loads data and splits it into train, validation and test sets.
    :param test_size (float): Value in [0, 1] indicating percentage of data set to allocate to test split
    :param validation_size (float): Value in [0, 1] indicating percentage of train set to allocate to validation split
    :return X_train (ndarray): Input training set
    :return X_validation (ndarray): Input validation set
    :return X_test (ndarray): Input test set
    :return y_train (ndarray): Target training set
    :return y_validation (ndarray): Target validation set
    :return y_test (ndarray): Target test set
    """

    # load data
    X, y = load_data(DATA_PATH)

    # create train, validation and test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=validation_size)

    # add an axis to input sets
    # X_train = X_train[..., np.newaxis]
    X_train = np.array(X_train)[:, np.newaxis, :, :]
    X_validation = np.array(X_test)[:, np.newaxis, :, :]
    X_test = np.array(X_test)[:, np.newaxis, :, :]

    return X_train, X_validation, X_test, y_train, y_validation, y_test


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # 1st conv layer
        self.cl1 = nn.Conv2d(in_channels=1, out_channels=32,  kernel_size=3)
        self.mp1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.batch1 = nn.BatchNorm2d(32)

        # 2nd conv layer
        self.cl2 = nn.Conv2d(32, 32, 3)
        self.mp2 = nn.MaxPool2d(3, stride=2, padding=1)
        self.batch2 = nn.BatchNorm2d(32)

        # 3rd conv layer
        self.cl3 = nn.Conv2d(32, 32, 2)
        self.mp3 = nn.MaxPool2d(2, stride=2, padding=1)
        self.batch3 = nn.BatchNorm2d(32)

        # flatten output and feed it into dense layer
        self.hl4 = nn.Linear(512, 64)

        # output layer
        self.hl5 = nn.Linear(64, 10)


    def forward (self, x):

        # 1st conv layer
        # print('input', x.shape)

        x = self.cl1(x)
        # print('conv1', x.shape)
        x = F.relu(x)
        # print('relu1', x.shape)
        x = self.mp1(x)
        # print('out1', x.shape)
        x = self.batch1(x)

        # x = F.batch_norm(x)
        # 2nd conv layer
        # print('input2', x.shape)
        x = self.cl2(x)
        # print('conv2', x.shape)
        x = F.relu(x)
        # print('relu2', x.shape)
        x = self.mp2(x)
        # print('out2', x.shape)
        x = self.batch2(x)

        # 3rd conv layer
        # print('input3', x.shape)
        x = self.cl3(x)
        # print('conv3', x.shape)
        x = F.relu(x)
        # print('relu3', x.shape)
        x = self.mp3(x)
        # print('out3', x.shape)
        x = self.batch3(x)

        x = torch.flatten(x, start_dim=1, end_dim=3)
        # print(x.shape)
        x = self.hl4(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)

        x = self.hl5(x)

        return x


def predict(model, X, y):

    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
    X = np.array(X)[ np.newaxis, :, :, :] # array shape (1, 130, 13, 1)
    X = Variable(torch.Tensor(X))


    # perform prediction
    with torch.no_grad():
        prediction = model(X)

    # get index with max value
    predicted_index = np.argmax(prediction, axis=1)

    print("Target: {}, Predicted label: {}".format(y, predicted_index))


def generate_dataloader(X, y, batch_size):

    X = torch.Tensor(X)
    y = torch.Tensor(y)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size)

    return loader


def train_model(model, train_loader):

    loss_list = []
    epoch_list = []

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(100):  # loop over the dataset multiple times

        loss_per_epoch = 0
        epoch_list.append(epoch)

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
            # print(inputs.shape)

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            loss_per_epoch += loss

            # print statistics
            running_loss += loss.data.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        loss_list.append(loss_per_epoch)

    print('Finished Training')
    return model, loss_list, epoch_list



def test_model(model, test_loader):

    correct = 0
    total = 0
    for data in test_loader:
        images, labels = data
        outputs = model(Variable(images))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()

    print('Accuracy of the network on the 100 songs: %d %%' % (
        100 * correct / total))
    return





# get train, validation, test splits
X_train, X_validation, X_test, y_train, y_validation, y_test = prepare_datasets(0.25, 0.2)


# create dataloader
train_loader = generate_dataloader(X_train, y_train, batch_size=32)
test_loader = generate_dataloader(X_test, y_test, batch_size=32)


# compile model
model = CNN()

# train model
trained_model, loss_list, epoch_list = train_model(model, train_loader)

plt.figure()
plt.plot(epoch_list, loss_list)
plt.show()
# plot accuracy/error for training and validation
#plot_history(history)

# evaluate model on test set
test = test_model(trained_model, test_loader)

# pick a sample to predict from the test set
X_to_predict = X_test[20]
y_to_predict = y_test[20]

# predict sample
predict(trained_model, X_to_predict, y_to_predict)