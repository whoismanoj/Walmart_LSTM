import numpy as np
import torch
import torch.nn as nn
import random

import matplotlib
import matplotlib.pyplot as plt

random.seed(0)
# Generate data with 52 features
timesteps = 1000
data = np.random.randint(low=0, high=100, size=(timesteps, 52))
#print(data)
#print(data.shape)
# Convert data to PyTorch tensors
data = torch.tensor(data, dtype=torch.float32)
#print(data)
#print(data.shape)
# Define LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

model = LSTMModel(52, 64, 52)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train model
losses = []
num_epochs = 3000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())


# Generate test data
test_data = np.random.randint(low=0, high=100, size=(20, 52))
test_data = torch.tensor(test_data, dtype=torch.float32)

# Plot the loss function
import matplotlib.pyplot as plt
plt.plot(losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

# Use model to make predictions on test data
predictions = model(test_data)

# Calculate MSE error
mse_error = nn.MSELoss()(predictions, test_data)
print('MSE error:', mse_error.item())

# Use model to make predictions
#test_predictions = model(test_data)
test_predictions = predictions.view(-1).detach().numpy()

# Calculate accuracy on test data
test_ground_truth = test_data.view(-1).numpy()

# Calculate the number of correct predictions
correct_predictions = np.sum(np.round(test_predictions) == test_ground_truth)

# Calculate the total number of predictions
total_predictions = len(test_predictions)

# Calculate the percentage of accuracy
accuracy = correct_predictions / total_predictions * 100
print('Test accuracy: {:.2f}%'.format(accuracy))


# Plot predictions against ground truth
plt.plot(test_data.view(-1).numpy(), label='Ground Truth')
plt.plot(test_predictions, label='Prediction')
plt.legend()
plt.show()

