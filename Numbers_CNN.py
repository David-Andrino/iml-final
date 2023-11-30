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
    print(f"    Shape of first image Tensor from current batch: {data[0].shape}")  ## Array shape: 28x28 1D
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
import math
# CNN MNIST classifier
class DigitClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()

        padding = 0
        dilation = 1
        stride = 1

        H = math.floor((28 + 2 * padding - dilation * (3 - 1) - 1)/stride + 1)
        
        for i in range(2):
            H = math.floor((H + 2*padding - dilation * (3 - 1) - 1)/stride + 1)
            print(str(i+2) + ": " + str(H))

 

        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3)),
            torch.nn.ReLU(),
            # Classification stage
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=H*H*128, out_features=10)        
        )
    
    def forward(self, x):
        return self.model(x)

cnn = DigitClassifier()
# for batch in train_batches:
#     X, y = batch
#     y_pred = cnn(X)
#     exit()

# TRAINING PROCEDURE
# Instance of the loss function and optimizer
opt = torch.optim.Adam(cnn.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()


# Number of times to pass through every batch
num_epochs = 10
# Set the CNN to training mode
cnn.train(True)
for epoch in range(num_epochs):
    for batch_idx, (X, y) in enumerate(train_batches):
        # Pass the batch of images to obtain a prediction
        y_pred = cnn(X)
        # Compute the loss comparing the predictions
        loss = loss_fn(input=y_pred, target=y)
        # Perform backpropagation with the computed loss
        # to compute the gradients
        loss.backward()
        # Update the weights with regard to the computed
        # gradients to minimize the loss
        opt.step()
        # In each iteration we want to compute new gradients,
        # that is why we set the gradients to 0
        opt.zero_grad()
        # Print to check the progress
        if batch_idx % 50 == 0:
            print(f"Train Epoch: {epoch} [{ batch_idx*len(X)}/{len(train_batches.dataset)} ({100.0 * batch_idx /len(train_batches):.0f}%)]\tLoss: {loss.item():.6f}")
    print(f"Epoch: {epoch} loss is {loss.item()}")


from torch import save, load
# Save the model state
with open("./cnn_model_state.pt", "wb") as file:
    save(cnn.state_dict(), file)

# Load the model state
with open("./cnn_model_state.pt", "rb") as file:
    cnn.load_state_dict(load(file))

