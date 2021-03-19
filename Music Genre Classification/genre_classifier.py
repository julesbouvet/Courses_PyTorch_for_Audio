import json
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.autograd import Variable
import torch.nn.functional as F


DATASET_PATH = "data.json"

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists into np.array
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    print(inputs.shape)

    return inputs, targets


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.hd1 = nn.Linear(1690, 512, bias=False)
        self.hd2 = nn.Linear(512, 256, bias=False)
        self.hd3 = nn.Linear(256, 64, bias=False)
        self.hd4 = nn.Linear(64, 10, bias=False)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1, end_dim=2)
        out = F.relu(self.hd1(x))
        out = F.relu(self.hd2(out))
        out = F.relu(self.hd3(out))
        out = self.hd4(out)

        return out




# load data
inputs, targets = load_data(DATASET_PATH)

# split the data into train and test sets
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.3)
inputs_train=torch.Tensor(inputs_train)
targets_train = torch.Tensor(targets_train)
inputs_test = torch.Tensor(inputs_test)
targets_test=torch.Tensor(targets_test)

train_dataset = TensorDataset(inputs_train, targets_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(inputs_test, targets_test)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

# model
model = Net()
print(model)


def train_model(model, train_loader):

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    for epoch in range(100):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.data.item()
            if i % 20 == 19:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')
    return model


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


trained_model=train_model(model, train_loader)

test= test_model(trained_model, test_loader)