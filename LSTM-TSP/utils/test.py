import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
input_size = 1
hidden_size = 50
output_size = 1
num_epochs = 200
learning_rate = 0.01

training_samples = 40
training_periods = 5
noise_period = 5
noise_amplitude = 1

# LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTM(input_size, hidden_size, output_size)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Generate a noisy sine wave
rng = np.random.default_rng()

x = np.sort(2 * training_periods * np.pi * rng.random((training_samples*training_periods, 1)), axis=0)
y = np.sin(x).ravel()
y[::noise_period] += noise_amplitude * (0.5 - rng.random(int(training_samples*training_periods/noise_period)))

# Convert to tensors
x_train = torch.tensor(y[:-1]).float().view(-1, 1, 1)
y_train = torch.tensor(y[1:]).float().view(-1, 1)

# Train the model
for epoch in range(num_epochs):
    outputs = model(x_train)
    optimizer.zero_grad()

    # Obtain the loss function
    loss = criterion(outputs, y_train)

    loss.backward()

    optimizer.step()

    if (epoch+1) % 10 == 0:
        print('Epoch: {}/{}, Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Test the model
y_pred = model(x_train)

# Plot the actual and predicted sine wave
plt.plot(y_train.detach().numpy(), label='actual')
plt.plot(y_pred.detach().numpy(), label='predicted')
plt.legend()
plt.show()
