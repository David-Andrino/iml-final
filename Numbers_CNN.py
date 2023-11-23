from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# Download the training data into the "data" folder
train = datasets.MNIST(root="data", download=True, train=True, transform=ToTensor())

# Group the training dataset into batches of size 32
train_batches = DataLoader(dataset=train, batch_size=32)

print(f"Number of created batches: {len(train_batches)}")

# Understand the data loaded
for batch in train_batches:

    # Check what is "batch"
    print(f"Type of 'batch': {type(batch)}")

    # Each batch returns the data (the images) and their corresponding labels
    data, labels = batch

    # Check what are "data" and "labels"
    print(f"    Type of 'data': {type(data)}")
    print(f"    Type of 'labels': {type(labels)}")
    # Check     the shape of "data" and "labels"
    print(f"    Shape of data Tensor from current batch: {data.shape}")
    print(f"    Shape of first image Tensor from current batch: {data[0].shape}")
    print(f"    Shape of labels Tensor from current batch: {labels.shape}")
    print(f"    Labels included in the 'labels' Tensor from current batch: {labels}")
    # Use "break" to exit the for loop
    break