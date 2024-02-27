import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
from collections import OrderedDict

class Net(nn.Module):
    def __init__(self, num_classes: int, num_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2 , 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        if num_channels == 3:
            self.fc1 = nn.Linear(16 * 5 * 5, 120)
        else:
            self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)#x = x.view(-1, 16 * 5 * 5) for CIFAR10  |||  x = x.view(-1, 16 * 4 * 4) if MNIST
        x = F.relu(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = self.fc3(x)
        return x
    
class LeNet(nn.Module):
    def __init__(self, num_classes: int, num_channels: int):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=1)
        return x
    
# This helps to generate initial parameters for the server so that the server does not need to request an initial parameter from a random client
def get_parameters(net) -> List[np.ndarray]:
    return [val.cpu().numpy() for _, val in net.state_dict().items()]

def train(net, trainloader, device, epochs, learning_rate, proximal_mu) -> None:
    # Train the network on the training set.
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate, weight_decay=0.001)
    global_params = [val.detach().clone() for val in net.parameters()]
    net.train()
    for _ in range(epochs):
        net = _train_one_epoch(net, global_params, trainloader, device, criterion, optimizer, proximal_mu)


def _train_one_epoch(net, global_params, trainloader, device, criterion, optimizer: torch.optim.Adam, proximal_mu: float) -> nn.Module:
    for images, labels in trainloader: 
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        proximal_term = 0.0
        for local_weights, global_weights in zip(net.parameters(), global_params):
            proximal_term += torch.square((local_weights - global_weights).norm(2))
        loss = criterion(net(images), labels) + (proximal_mu / 2) * proximal_term
        loss.backward()
        optimizer.step()
    return net


def test(net, testloader, device: str):
    # Validate the network on the entire test set, and report loss and accuracy.

    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    net.to(device)
    with torch.no_grad():
        for data in testloader:
            images, labels = data[0].to(device), data[1].to(device)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)

    return loss, accuracy