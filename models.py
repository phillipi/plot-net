import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        self.l1 = torch.nn.Linear(3,2,bias=False)
        self.relu1 = nn.ReLU()
        self.l2 = torch.nn.Linear(3,2,bias=False)
        self.relu2 = nn.ReLU()
        self.l3 = torch.nn.Linear(3,2,bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.layer_names = ['linear', 'relu', 'linear', 'relu', 'linear', 'softmax']

        self.s = torch.tensor([1.0],requires_grad=True)

    def forward(self, x):

        output = x.view(1, x.size(0), x.size(1))
        for layer in net.children():
            if isinstance(layer, nn.Linear):
                input = torch.cat([output[-1],torch.ones([x.size(0),1])],dim=1) # cat for bias
            else:
                input = output[-1]
            new_output = layer(input).view(1, input.size(0), x.size(1))
            output = torch.cat((output,new_output), axis=0)

        return output

    def forward_grid(self, x):
        # applies each layer separately to the input x, rather than in sequence

        output = x.view(1, x.size(0), x.size(1))
        for layer in net.children():
            if isinstance(layer, nn.Linear):
                input = torch.cat([x,torch.ones([x.size(0),1])],dim=1) # cat for bias
            else:
                input = x
            new_output = layer(input).view(1, input.size(0), x.size(1))
            output = torch.cat((output,new_output), axis=0)

        return output

    def get_layer_name(self, l):
        return self.layer_names[l]

def weights_init(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()
        torch.nn.init.eye_(m.weight.data)
        if m.bias is not None:
          torch.nn.init.zeros_(m.bias)

def computeLoss(net, criterion, X, Y):
  output = net(X)
  loss = criterion(torch.log(output[-1]), Y)
  return loss

def trainForABit(net, criterion, optimizer, X, Y, N_train_iter_per_viz):
  for jj in range(N_train_iter_per_viz):
      optimizer.zero_grad()
      loss = computeLoss(net,criterion,X,Y)
      loss.backward()
      optimizer.step()
  return loss
