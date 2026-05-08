import pyro
from pyro.nn import PyroModule, PyroSample
import pyro.distributions as dist
import torch
import torch.nn as nn

class BNN_onestep_cnn(PyroModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = PyroModule[nn.Conv1d](in_channels = 3, out_channels = 6, kernel_size = 2, padding = 'same') # First convolutional layer
        self.activation = PyroModule[nn.ReLU]() #ReLU
        self.layer_2 = PyroModule[nn.Conv1d](6, 12, 2, padding='same') # Second convolutional layer
        self.linear_1 = PyroModule[nn.Linear](12,1) # Fully connected layer with an output of one

        self.layer_1.weight = PyroSample(dist.Normal(0, 1).expand([6,3,2]).to_event(3)) # Convolutional layer weight prior with a standard normal distribution as prior
        self.layer_1.bias = PyroSample(dist.Normal(0, 1).expand([6]).to_event(1)) # Convolutional layer bias prior with a standard normal distribution as prior
        self.layer_2.weight = PyroSample(dist.Normal(0, 1).expand([12,6,2]).to_event(3))
        self.layer_2.bias = PyroSample(dist.Normal(0, 1).expand([12]).to_event(1))
        self.linear_1.weight = PyroSample(dist.Normal(0, 1).expand([1,12]).to_event(2))
        self.linear_1.bias = PyroSample(dist.Normal(0, 1).expand([1]).to_event(1))

    def forward(self, x, y = None):

        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        x = torch.flatten(x,1)
        x = self.activation(self.linear_1(x))
        mu = x
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1)) # Gamma prior for the standard deviation

        if y is not None:
            y = torch.flatten(y,1)

        with pyro.plate("data", x.shape[0]):

            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma).to_event(1), obs = y) # Update the Posterior
        return mu


class BNN_multistep_cnn(PyroModule):
    def __init__(self):
        super().__init__()
        self.layer_1 = PyroModule[nn.Conv1d](in_channels = 3, out_channels = 6, kernel_size = 2, padding = 'same') # First convolutional layer
        self.activation = PyroModule[nn.ReLU]() # ReLU
        self.layer_2 = PyroModule[nn.Conv1d](6, 12, 2, padding='same') # Second convolutional layer
        self.linear_1 = PyroModule[nn.Linear](12,3) # Fully connected layer with output of three

        self.layer_1.weight = PyroSample(dist.Normal(0, 1).expand([6,3,2]).to_event(3)) # Convolutional layer weight prior with a standard normal distribution as prior
        self.layer_1.bias = PyroSample(dist.Normal(0, 1).expand([6]).to_event(1)) # Convolutional layer bias prior with a standard normal distribution as prior
        self.layer_2.weight = PyroSample(dist.Normal(0, 1).expand([12,6,2]).to_event(3))
        self.layer_2.bias = PyroSample(dist.Normal(0, 1).expand([12]).to_event(1))
        self.linear_1.weight = PyroSample(dist.Normal(0, 1).expand([3,12]).to_event(2))
        self.linear_1.bias = PyroSample(dist.Normal(0, 1).expand([3]).to_event(1))

    def forward(self, x, y = None):

        x = self.activation(self.layer_1(x))
        x = self.activation(self.layer_2(x))
        x = torch.flatten(x,1)
        x = self.activation(self.linear_1(x))
        mu = x
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1)) # Gamma prior for the standard deviation

        if y is not None:
            y = torch.flatten(y,1)

        with pyro.plate("data", x.shape[0]):

            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma).to_event(1), obs = y) # Update the Posterior
        return mu


class BNN_linear_onestep(PyroModule):
    def __init__(self):
        super().__init__()
        self.linear_1 = PyroModule[nn.Linear](3,10) # First linear layer
        self.activation = PyroModule[nn.ReLU]() # ReLU
        self.linear_2 = PyroModule[nn.Linear](10,30) # Second linear layer
        self.linear_3 = PyroModule[nn.Linear](30,1) # Third linear layer with an output of one

        self.linear_1.weight = PyroSample(dist.Normal(0, 1).expand([10,3]).to_event(2)) # Linear layer weight prior with a standard normal distribution as prior
        self.linear_1.bias = PyroSample(dist.Normal(0, 1).expand([10]).to_event(1)) # Linear layer bias prior with a standard normal distribution as prior
        self.linear_2.weight = PyroSample(dist.Normal(0, 1).expand([30,10]).to_event(2))
        self.linear_2.bias = PyroSample(dist.Normal(0, 1).expand([30]).to_event(1))
        self.linear_3.weight = PyroSample(dist.Normal(0, 1).expand([1,30]).to_event(2))
        self.linear_3.bias = PyroSample(dist.Normal(0, 1).expand([1]).to_event(1))

    def forward(self, x, y = None):

        x = torch.flatten(x,1)
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        mu = x
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1)) # Gamma prior for the standard deviation

        if y is not None:
            y = torch.flatten(y,1)

        with pyro.plate("data", x.shape[0]):

            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma).to_event(1), obs = y) # Update the posterior
        return mu


class BNN_linear_multistep(PyroModule):
    def __init__(self):
        super().__init__()
        self.linear_1 = PyroModule[nn.Linear](3,10) # First linear layer
        self.activation = PyroModule[nn.ReLU]() # ReLU
        self.linear_2 = PyroModule[nn.Linear](10,30) # Second linear layer
        self.linear_3 = PyroModule[nn.Linear](30,3) # Third linear layer with an output of three

        self.linear_1.weight = PyroSample(dist.Normal(0, 1).expand([10,3]).to_event(2)) # Linear layer weight prior with a standard normal distribution as prior
        self.linear_1.bias = PyroSample(dist.Normal(0, 1).expand([10]).to_event(1)) # Linear layer bias prior with a standard normal distribution as prior
        self.linear_2.weight = PyroSample(dist.Normal(0, 1).expand([30,10]).to_event(2))
        self.linear_2.bias = PyroSample(dist.Normal(0, 1).expand([30]).to_event(1))
        self.linear_3.weight = PyroSample(dist.Normal(0, 1).expand([3,30]).to_event(2))
        self.linear_3.bias = PyroSample(dist.Normal(0, 1).expand([3]).to_event(1))

    def forward(self, x, y = None):

        x = torch.flatten(x,1)
        x = self.activation(self.linear_1(x))
        x = self.activation(self.linear_2(x))
        x = self.activation(self.linear_3(x))
        mu = x
        sigma = pyro.sample("sigma", dist.Gamma(.5, 1)) # Gamma prior for the standard deviation

        if y is not None:
            y = torch.flatten(y,1)

        with pyro.plate("data", x.shape[0]):

            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma).to_event(1), obs = y) # Update the posterior
        return mu


