import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def orthogonal_init(layer, gain = 1.0):
    for name, param in layer.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(param, gain = gain)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
        else:
            raise NotImplementedError
    
class Q_network_RNN(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_RNN, self).__init__()
        self.args = args
        self.rnn_hidden = None

        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)    

    

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        Q = self.fc2(self.rnn_hidden)
        return Q
        
        
class Q_network_MLP(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network_MLP, self).__init__()
        self.args = args

        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        self.hidden = F.relu(self.fc2(x))
        Q = self.fc3(self.hidden)
        return Q
            
    

    
class V_network_RNN(nn.Module):
    def __init__(self, args, input_dim) -> None:
        super(V_network_RNN, self).__init__()
        self.rnn_hidden = None
        self.fc1 = nn.Linear(input_dim, args.rnn_hidden_dim)
        self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, 1)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.rnn)
            orthogonal_init(self.fc2)


    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        self.rnn_hidden = self.rnn(x, self.rnn_hidden)
        V = self.fc2(self.rnn_hidden)

        return V

class Q_jt_network_MLP(nn.Module):
    def __init__(self, args, input_dim, seed) -> None:
        super(Q_jt_network_MLP, self).__init__()
        self.args = args
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        Q_jt = self.fc3(x)
        return Q_jt


    
class V_jt_network_MLP(nn.Module):
    def __init__(self, args, input_dim, seed) -> None:
        super(V_jt_network_MLP, self).__init__()
        self.args = args
        torch.manual_seed(seed)
        self.fc1 = nn.Linear(input_dim, args.mlp_hidden_dim)
        self.fc2 = nn.Linear(args.mlp_hidden_dim, args.mlp_hidden_dim)
        self.fc3 = nn.Linear(args.mlp_hidden_dim, 1)

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        V_jt = self.fc3(x)
        return V_jt
    
