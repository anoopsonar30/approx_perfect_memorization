import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

torch.manual_seed(43)

num_datapoints = 5
x_lim = int((num_datapoints - 1) * 0.5)

# 1. Generate the dataset with random points
x = torch.linspace(-x_lim, x_lim, steps=num_datapoints).T.unsqueeze(1)
x_unknown_points = torch.linspace(-x_lim + 0.5, x_lim + 0.5, steps=num_datapoints).T.unsqueeze(1)
x = torch.cat((x, x_unknown_points), 0)

y = torch.zeros(num_datapoints, 1) # + 0.2 * torch.normal(torch.zeros(x.size()))  # y = x with some noise
num_outliers = num_datapoints # int(num_datapoints / 10)
# outlier_x = torch.randint(0, num_datapoints, (num_outliers, 1))
for i in range(num_outliers):
    # y[i] = x[i] + 50 * torch.normal(torch.zeros(1))
    y[i] = 5 * torch.normal(torch.zeros(1)) # Overwrite y[i] with a random value
y_unknown_points = torch.zeros(num_datapoints, 1)
y = torch.cat((y, y_unknown_points), 0)

# 2. Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()        
        self.fc1 = nn.Linear(1, num_datapoints)
        self.fc2 = nn.Linear(num_datapoints, 1)

    def forward(self, x):
        
        sd = 0.05
        
        # Intuitively, it seems that a gaussian activation function with a sufficiently small standard deviation should nudge the network
        # to almost perfectly memorize the training data similar to a key-value map and in this case return zero elsewhere.
        # One can easily hand design the weights for this architecture such that the loss is zero, but the 
        # optimizer is NOT able to find that solution likely because the problem is non-convex.
        # QUESTION  : Can we design an inductive bias that will allow us to (almost) perfectly memorize the training data?
        #             and return zero elsewhere (for this scenario)?
        
        # Perhaps the gaussian activation has a slope too close to zero at points far from the training data?
        # Initially I considered using a finite-width dirac-delta function with a very small width as a possible activation function
        # but in that case the gradient is zero everywhere except at the points where it's discontinuous.
        # I have a feeling that there might be a way to leverage something from distribution theory stuff to make this work.
        
        # EXPECTED RESULT : The final plot we would like to see is one where the red line is mostly flat with a few sharp spikes at the
        # training data points where the y-values are non-zero.
        
        x = self.fc1(x)
        x = torch.exp(-x**2 / (2 * sd**2)) # Gaussian activation function
        x = self.fc2(x)

        return x

net = Net()

# 3. Define a loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(net.parameters(), lr=5e-2, momentum=0.2)
 
# 4. Train the model
for epoch in range(10000):
    optimizer.zero_grad()
    outputs = net(x)
                
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    
    print(loss)
    
num_plotpoints = 11
plot_lim = int((num_plotpoints - 1) * 0.5)
# 5. Evaluate the model and plot
test_x = torch.linspace(-plot_lim, plot_lim, steps=num_plotpoints).T.unsqueeze(1) 
with torch.no_grad():
    predicted = net(test_x)

# Plotting
plt.scatter(x.data.numpy(), y.data.numpy(), label='Original Data', color='blue')
plt.plot(test_x.data.numpy(), predicted, label='Predictions', color='red')
plt.legend()
plt.show()
