from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download the training data into the "data" folder
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())

# Group the training dataset into batches of size 32
train_batches = DataLoader(dataset=train, batch_size=32)

''' Exercise 1
print(f"Number of created batches: {len(train_batches)}")
for batch in train_batches:
    print(f"Type of 'batch': {type(batch)}")
    data, labels = batch
    print(f"    Type of 'data': {type(data)}")
    print(f"    Type of 'labels': {type(labels)}")
    print(f"    Shape of data Tensor from current batch: {data.shape}")
    print(f"    Shape of first image Tensor from current batch: {data[0].shape}")
    print(f"    Shape of labels Tensor from current batch: {labels.shape}")
    print(f"    Labels included in the 'labels' Tensor from current batch: {labels}")
    break
'''

''' Draw training images
import matplotlib.pyplot as plt

for batch in train_batches:
    plt.figure()
    data, labels = batch
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(data[i][0], cmap="gray", interpolation="none")
        plt.title(f"Ground Truth label: {labels[i]}")
        plt.xticks([])
        plt.yticks([])

    plt.show()
    plt.close()
    break
'''

# Define the Convolutional Neural Network architecture
import torch
# CNN MNIST classifier
class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2D()
        )

    def forward(self, x):
        return self.model(x)
