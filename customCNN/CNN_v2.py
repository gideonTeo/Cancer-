import torch.nn as nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.9)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.1)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(num_features=out_channels, eps=1e-05, momentum=0.9)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.1)
        self.pool = nn.MaxPool2d(kernel_size=(2,2))
        self.dropout = nn.Dropout(p=0.25)

    def forward(self, input):
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.leaky_relu1(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.leaky_relu2(output)
        output = self.pool(output)
        output = self.dropout(output)

        return output


class CNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN, self).__init__()

        self.block1 = Block(in_channels=3, out_channels=16)
        self.block2 = Block(in_channels=16, out_channels=32)
        self.block3 = Block(in_channels=32, out_channels=64)
        self.block4 = Block(in_channels=64, out_channels=128)
        self.block5 = Block(in_channels=128, out_channels=256)
        self.block6 = Block(in_channels=256, out_channels=512)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512*3*3, 1024)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(1024, 512)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc3 = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input):
        output = self.block1(input)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.block6(output)
        
        output = self.flatten(output)
        
        output = self.fc1(output)
        output = self.relu1(output)
        output = self.dropout1(output)
        output = self.fc2(output)
        output = self.relu2(output)
        output = self.dropout2(output)
        output = self.fc3(output)
        output = self.softmax(output)
        return output