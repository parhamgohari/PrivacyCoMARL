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
        
class Encoder(nn.Module):
    def __init__(self, args, input_dim):
        super(Encoder, self).__init__()
        self.args = args
        self.hidden_dim = args.rnn_hidden_dim if args.use_rnn else args.hidden_dim
        self.fc1 = nn.Linear(input_dim, self.hidden_dim)
        if args.use_rnn:
            self.hidden = None
            self.rnn = nn.GRU(self.hidden_dim, self.hidden_dim, batch_first = True)
        else:
            self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            if args.use_rnn:
                orthogonal_init(self.rnn)
            else:
                orthogonal_init(self.fc2)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        if self.args.use_rnn:
            out, self.hidden = self.rnn(x, self.hidden)
            return out
        else:
            out = F.relu(self.fc2(x))
            return out
        
class Q_network(nn.Module):
    def __init__(self, args, input_dim):
        super(Q_network, self).__init__()
        self.args = args
        self.hidden_dim = args.rnn_hidden_dim if args.use_rnn else args.hidden_dim
        self.encoder = Encoder(args, input_dim)
        self.fc = nn.Linear(self.hidden_dim, args.action_dim)

        if args.use_orthogonal_init:
            orthogonal_init(self.fc)    

    def forward(self, inputs):
        if self.args.use_rnn:
            Q = self.fc(self.encoder(inputs))
        else:
            Q = self.fc(F.relu(self.encoder(inputs)))
        return Q
    

            
    

        