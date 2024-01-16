import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

num_datapoints = 50

# 1. Generate the dataset with random points
x = torch.rand(num_datapoints, 1) * 200 - 100  # Random points in [-100, 100]
y = x # + 0.2 * torch.normal(torch.zeros(x.size()))  # y = x with some noise

# 2. Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

net = Net()

# 3. Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=0.0001)

# 4. Train the model
for epoch in range(100):
    optimizer.zero_grad()
    outputs = net(x)
                
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
num_plotpoints = 100
# 5. Evaluate the model and plot
test_x = torch.rand(num_plotpoints, 1) * 200 - 100  # Random test points
with torch.no_grad():
    predicted = net(test_x)

# Plotting
plt.scatter(x.data.numpy(), y.data.numpy(), label='Original Data', color='blue')
plt.plot(test_x.data.numpy(), predicted, label='Predictions', color='red')
plt.legend()
plt.show()
