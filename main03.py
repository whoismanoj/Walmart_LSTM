import torch
import torch.nn as nn
import numpy as np

# Generate training data
timesteps = 1000
data = np.sin(np.linspace(0, 10*np.pi, timesteps))
#print(data)
# Reshape data for LSTM input
data = torch.tensor(data, dtype=torch.float32).view(1, timesteps, 1)
print(data.shape)

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

model = LSTMModel(1, 50, 1)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

# Train model
num_epochs = 500
for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data)
    loss.backward()
    optimizer.step()

# Generate test data
test_timesteps = 500
test_data = np.sin(np.linspace(10*np.pi, 12*np.pi, test_timesteps))
test_data = torch.tensor(test_data, dtype=torch.float32).view(1, test_timesteps, 1)

# Use model to make predictions
predictions = model(test_data)
predictions = predictions.view(-1).detach().numpy()

# Plot predictions against ground truth
import matplotlib.pyplot as plt
plt.plot(test_data.view(-1).numpy(), label='Ground Truth')
plt.plot(predictions, label='Prediction')
plt.legend()
plt.show()