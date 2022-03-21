- CV challenge Repo: https://github.com/udacity/CVND_Exercises/tree/master/1_5_CNN_Layers
- [compare pytorch vs tensorflow](https://towardsdatascience.com/pytorch-vs-tensorflow-1-month-summary-35d138590f9)
- [pytorch doc](https://pytorch.org/docs/stable/nn.html)
## Data Loading
[tutorial](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)

## Module
During training, PyTorch will be able to perform backpropagation by keeping track of the network's feedforward behavior and using autograd to calculate the update to the weights in the network.
```
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self, n_classes):
        super(Net, self).__init__()

        # 1 input image channel (grayscale), 32 output channels/feature maps
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)

        # maxpool layer
        # pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)

        # fully-connected layer
        # 32*4 input size to account for the downsampled image size after pooling
        # num_classes outputs (for n_classes of image data)
        self.fc1 = nn.Linear(32*4, n_classes)

    # define the feedforward behavior
    def forward(self, x):
        # one conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))

        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1) # the size -1 is inferred from other dimensions
        # linear layer
        x = F.relu(self.fc1(x))

        # final output
        return x

# instantiate and print your Net
n_classes = 20 # example number of classes
net = Net(n_classes)
print(net)
```
## Loss and Optimizer
https://pytorch.org/docs/master/optim.html <br>
https://pytorch.org/docs/master/nn.html#loss-functions
- Prepares all input images and label data for training
- Passes the input through the network (forward pass)
- Computes the loss (how far is the predicted classes are from the correct labels)
- Propagates gradients back into the network’s parameters (backward pass)
- Updates the weights (parameter update)

### flattern:
```
x = torch.randn(1, 2, 3)
tensor([[[ 0.3982,  0.5383,  0.6959],
         [-0.9727, -0.5536,  1.1395]]])
x.view(x.size(0), -1)
tensor([[ 0.3982,  0.5383,  0.6959, -0.9727, -0.5536,  1.1395]])

```
### unsqueeze
Returns a new tensor with a dimension of size one inserted at the specified position.

```
x = torch.tensor([1, 2, 3, 4])
tensor([1, 2, 3, 4])
torch.unsqueeze(x, 1)
tensor([[1],
        [2],
        [3],
        [4]])
```
