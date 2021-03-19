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


    return X_train, X_validation, X_test, y_train, y_validation, y_test


class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()

        # 1st LSTM layer
        self.lstm1 = nn.LSTM(13, 64, 1)

        # 2nd LSTM layer
        self.lstm2 = nn.LSTM(64, 64)

        # 1st dense layer
        self.hl3 = nn.Linear(8320, 64)

        # output layer
        self.hl4 = nn.Linear(64, 10)



    def forward (self, x):

        # 1st LSTM layer
        x, (hn,cn) = self.lstm1(x)


        # 2nd LSTM layer
        x, (hn, cn) = self.lstm2(x)

        # 1st dense layer
        x = torch.flatten(x, 1, 2)
        x = self.hl3(x)
        x = F.relu(x)
        x = F.dropout(x, 0.3)

        # output layer
        x = self.hl4(x)

        return x


def predict(model, X, y):

    """Predict a single sample using the trained model
    :param model: Trained classifier
    :param X: Input data
    :param y (int): Target
    """

    # add a dimension to input data for sample - model.predict() expects a 4d array in this case
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

# plot accuracy/error for training and validation
plt.figure()
plt.plot(loss_list, epoch_list)
plt.show()


# evaluate model on test set
test = test_model(trained_model, test_loader)

# pick a sample to predict from the test set
X_to_predict = X_test[20]
y_to_predict = y_test[20]

# predict sample
predict(trained_model, X_to_predict, y_to_predict)