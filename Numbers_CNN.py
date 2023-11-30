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

from math import floor

# PARAMETER CALCULATION
out = [64, 128, 256]
kernels = [(3, 3), (3, 3), (3, 3)]

padding = 0
dilation = 1
stride = 1

H = 28
train = True # True for training the CNN, False for loading from disk
evaluate = False # True for running the prediction on the whole database and calculate accuracy
img_idx = 111 # If evaluate is false, predict this single image

for i in range(0, 3):
    H = floor((H + 2*padding - dilation*(kernels[i][0]-1) - 1)/stride + 1)

in_features = out[2]*H*H
num_epochs = 1

# PREDICTION
import torch
class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(
                in_channels=1, 
                out_channels=out[0], 
                kernel_size=kernels[0]
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out[0],
                out_channels=out[1],
                kernel_size=kernels[1]
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                in_channels=out[1],
                out_channels=out[2],
                kernel_size=kernels[2]
            ),
            torch.nn.ReLU(),
            # Classification stage
            torch.nn.Flatten(),
            torch.nn.Linear(in_features, out_features=10)
        )

    def forward(self, x):
        return self.model(x)

cnn = DigitClassifier()

if train:
    opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    cnn.train(True)

    for epoch in range(num_epochs):
        for batch_idx, (X, y) in enumerate(train_batches):
            y_pred = cnn(X)
            loss = loss_fn(input=y_pred, target = y)
            loss.backward()
            opt.step()
            opt.zero_grad()

            if batch_idx % 50 == 0:
                print(
                    f"Train Epoch: {epoch} [{ batch_idx *len(X)}/{len(train_batches.dataset)} ({100.0 * batch_idx / len(train_batches):.0f}%)]\tLoss: {loss.item():.6f}"
                )
        print(f"Epoch: {epoch} loss is {loss.item()}")
    
    with open("./cnn_model_state.pt", "wb") as f:
        torch.save(cnn.state_dict(), f)

else:
    with open("./cnn_model_state.pt", "rb") as f:
        cnn.load_state_dict(torch.load(f))

# TEST DATA
test_data = datasets.MNIST(root="data", download=True, train=False, transform=ToTensor())
test_batches = DataLoader(dataset=test_data, batch_size=32)
cnn.eval()

if evaluate:
    with torch.no_grad(): # Avoid calculating gradients
        # Predict the images in batches
        for images, labels in test_batches:
            # Predict the images with probabilities
            test_output = cnn(images)
            # For each image, take the highest probability
            y_test_pred = torch.max(test_output, 1)[1].data.squeeze()
            # Measure the accuracy
            accuracy = (y_test_pred == labels).sum().item() / float(labels.size(0))
            print("VALIDATION SET ACCURACY: %.2f" % accuracy)
else:
    img_tensor, label = test_data[img_idx]
    print(f"Shape of the tensor: {img_tensor.shape}")

    img_tensor = img_tensor.unsqueeze(0)
    print(f"Shape of the tensor: {img_tensor.shape}")