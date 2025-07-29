import torch
import torch.nn.functional as F
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, d=2):
        super(Net, self).__init__()
        self.criterion = None
        self.d = d

    @property
    def layer_names(self):
        return [layer.__class__.__name__.lower() for layer in self.children()]

    @property
    def get_n_layers(self):
        return len(self.layer_names)
    
    def get_layer_name(self, layer_idx):
        return self.layer_names[layer_idx]

    def forward(self, x):
        """
        Apply each layer of the network in sequence
        Returns: [x, f1(x), f2(f1(x)), ...], where fi is layer i
        """
        output = x.view(1, x.size(0), x.size(1)) # list of embeddings per layer; first one is just the input data
        for layer in self.children():
            input = output[-1]
            new_output = layer(input).view(1, input.size(0), x.size(1))
            output = torch.cat((output,new_output), axis=0)

        return output

    def forward_nonsequential(self, x):
        """
        Apply each layer of the network separately to the same grid of input points
        Returns: [x, f1(x), f2(x), ...], where fi is layer i
        """
        output = x.view(1, x.size(0), x.size(1)) # list of embeddings per layer; first one is just the input data
        input = x
        for layer in self.children():
            new_output = layer(input).view(1, input.size(0), x.size(1))
            output = torch.cat((output,new_output), axis=0)

        return output

    def weights_init(self, m):
        # init to identity for clearer visualization
        if isinstance(m, torch.nn.Linear):
            m.reset_parameters()
            torch.nn.init.eye_(m.weight.data)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def loss(self, X, Y, criterion):
        output = self.forward(X)
        if isinstance(criterion, torch.nn.NLLLoss):
            loss = criterion(torch.log(output[-1]), Y)
        else:
            loss = criterion(output[-1], Y)
        return loss

# Example simple network: an MLP with width d
class MySimpleNet(Net):
    def __init__(self, d=2):
        super(MySimpleNet, self).__init__(d)

        Nx = d # data dimensionality
        Nz = d # width
        Ny = d # output dimensionality

        self.l1 = torch.nn.Linear(Nx, Nz)
        self.relu1 = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(Nz, Nz)
        self.relu2 = torch.nn.ReLU()
        self.l3 = torch.nn.Linear(Nz, Ny)
        self.softmax = torch.nn.Softmax(dim=1)

        self.apply(self.weights_init)


# Example simple resnet
class ResidualBlock(torch.nn.Module):
    def __init__(self, d):
        super(ResidualBlock, self).__init__()
        self.linear = torch.nn.Linear(d, d)
        self.relu = torch.nn.ReLU()
    
    def forward(self, x):
        residual = x
        out = self.linear(x)
        out = self.relu(out)
        return out + 0.1*residual

class SimpleResnet(Net):
    def __init__(self, d=2):
        super(SimpleResnet, self).__init__(d)

        self.residual1 = ResidualBlock(d)
        self.residual2 = ResidualBlock(d)
        self.residual3 = ResidualBlock(d)
        self.residual4 = ResidualBlock(d)
        self.l1 = torch.nn.Linear(d, d)
        self.softmax = torch.nn.Softmax(dim=1)

        self.apply(self.weights_init)


class LinearLayer(Net):
    def __init__(self, d=2):
        super(LinearLayer, self).__init__(d)

        Nx = d # data dimensionality
        Nz = d # width

        self.linear = torch.nn.Linear(Nx, Nz)

        self.apply(self.weights_init)
    
    def weights_init(self, m):
        if self.d == 2:
            # for 2D case, init to a nice simple rotation + translation, for visualization
            if isinstance(m, torch.nn.Linear):
                angle = np.pi/5
                m.weight.data = torch.FloatTensor([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
                m.bias.data = torch.FloatTensor([-0.5,0.5])
        elif self.d == 3:
            # for 3D case, init to a small rotation around z and a translation
            if isinstance(m, torch.nn.Linear):
                angle = np.pi/10
                rot = torch.FloatTensor([
                    [np.cos(angle), -np.sin(angle), 0],
                    [np.sin(angle),  np.cos(angle), 0],
                    [0,              0,             1]
                ])
                m.weight.data = rot
                m.bias.data = torch.FloatTensor([-0.25, 0.5, 0.5])
        else:
            # init to identity for clearer visualization
            if isinstance(m, torch.nn.Linear):
                m.reset_parameters()
                torch.nn.init.eye_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

class DiffusionLayer(torch.nn.Module):
    def __init__(self, d=2):
        super(DiffusionLayer, self).__init__()
        self.d = d

    def forward(self, x):
        return 0.92*x+torch.randn_like(x)*0.08

class DiffusionNet(Net):
    def __init__(self, d=2):
        super(DiffusionNet, self).__init__(d)

        self.diffusion_layer1 = DiffusionLayer(d)
        self.diffusion_layer2 = DiffusionLayer(d)
        self.diffusion_layer3 = DiffusionLayer(d)
        self.diffusion_layer4 = DiffusionLayer(d)
        
class Norm(torch.nn.Module):
    def __init__(self,p):
        super(Norm, self).__init__()
        self.p = p

    def forward(self, x):
        return F.normalize(x, self.p, dim=1)

class L2NormLayer(Net):
    def __init__(self, d=2):
        super(L2NormLayer, self).__init__(d)

        self.l2_norm = Norm(2)

class ReLULayer(Net):
    def __init__(self, d=2):
        super(ReLULayer, self).__init__(d)

        self.relu = torch.nn.ReLU()

class SoftmaxLayer(Net):
    def __init__(self, d=2):
        super(SoftmaxLayer, self).__init__(d)

        self.softmax = torch.nn.Softmax(dim=1)

def mk_model(which_model, d=2):
    if which_model == 'MySimpleNet':
        return MySimpleNet(d)
    elif which_model == 'SimpleResnet':
        return SimpleResnet(d)
    elif which_model == 'linear':
        return LinearLayer(d)
    elif which_model == 'diffusion':
        return DiffusionNet(d)
    elif which_model == 'relu':
        return ReLULayer(d)
    elif which_model == 'l2_norm':
        return L2NormLayer(d)
    elif which_model == 'softmax':
        return SoftmaxLayer(d)
    else:
        raise ValueError(f"Model {which_model} not found")