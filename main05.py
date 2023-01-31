import numpy as np
import torch
import torch.nn as nn
from datasource import get_data_random
from datasource import get_data_sine_wave
from datasource import get_dataset_5

# Generate random dataset
np.random.seed(0)
timesteps = 281 # this is the length of the dataset. its a total number of data point.
data = get_dataset_5() # it is a real dataset downladed from UCI
print(data)


# Split data into train and test sets
train_data = data[:int(timesteps*0.8)]
test_data = data[int(timesteps*0.8):]

# Convert data to PyTorch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)
test_data = torch.tensor(test_data, dtype=torch.float32)

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

model = LSTMModel(input_size=1, hidden_size=64, output_size=1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train model
# Keep track of losses during training
test_accuracies = []
losses = []
num_epochs = 50000
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(train_data)
    loss = criterion(output, train_data)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())


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

