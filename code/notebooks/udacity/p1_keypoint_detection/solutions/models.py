## v2: added conv5 on top of v1, batch=20, momenturm =0.9, lr=0.01

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, debug=False):
        super(Net, self).__init__()
        dp=0.1
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 96, 3)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(96, 128, 3)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.conv5 = nn.Conv2d(128, 256, 3)
        self.pool5 = nn.MaxPool2d(2, 2)
        self.debug = debug
        self.conv1_drop = nn.Dropout(p=dp)
        self.conv2_drop = nn.Dropout(p=dp)
        self.conv3_drop = nn.Dropout(p=dp)
        self.conv4_drop = nn.Dropout(p=dp)
        self.conv5_drop = nn.Dropout(p=dp)
        
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        num_in=4096
        num_mid = 1024
        num_out = 2 * 68
        #test_sample['image'].shape[2:]
        self.fc1 = nn.Linear(num_in, num_mid)
#         self.fc1_drop = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(num_mid, num_out)
        
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.conv1_drop(x)
        if self.debug:
            print('after pool1: ', x.shape)
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.conv2_drop(x)
        if self.debug:
            print('after pool2: ', x.shape)
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.conv3_drop(x)
        if self.debug:
            print('after pool3: ', x.shape)
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.conv4_drop(x)
        if self.debug:
            print('after pool4: ', x.shape)
        x = self.pool5(F.relu(self.conv5(x)))
        x = self.conv5_drop(x)
        if self.debug:
            print('after pool5: ', x.shape)
        x = x.view(x.size(0), -1)
        if self.debug:
            print('after view: ', x.shape)
        x = F.relu(self.fc1(x))
        if self.debug:
            print('after fc1: ', x.shape)
        x = self.fc2(x)
        if self.debug:
            print('after fc2: ', x.shape)
        x = F.tanh(x)
#         x = torch.mul(x, 5)
        if self.debug:
            print('after tanh: ', x.shape)
        return x

