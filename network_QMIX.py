import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers.dp_rnn import DPGRUCell

def orthogonal_init(layer, gain = 1.0):
    for name, param in layer.named_parameters():
        if 'weight' in name:
            nn.init.orthogonal_(param, gain = gain)
        elif 'bias' in name:
            nn.init.constant_(param, 0.0)
        else:
            raise NotImplementedError
        
class Encoder(nn.Module):
    def __init__(self, args, input_dim, output_dim):
        super(Encoder, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_dim, args.encoder_hidden_dim)
        if args.use_rnn:
            self.hidden = None
            self.rnn = nn.GRU(args.encoder_hidden_dim, output_dim, batch_first = True)
        else:
            self.fc2 = nn.Linear(args.encoder_hidden_dim, output_dim)
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
        self.encoder = Encoder(args, input_dim, args.q_network_hidden_dim)
        self.fc = nn.Linear(args.q_network_hidden_dim, args.action_dim)

        if args.use_orthogonal_init:
            orthogonal_init(self.fc)    

    def forward(self, inputs):
        if self.args.use_rnn:
            Q = self.fc(self.encoder(inputs))
        else:
            Q = self.fc(F.relu(self.encoder(inputs)))
        return Q
    
class Vmix_Net(nn.Module):
    def __init__(self, args, input_dim) -> None:
        super(Vmix_Net, self).__init__()
        self.args = args
        self.fc1 = nn.Linear(input_dim, args.v_network_hidden_dim)
        self.fc2 = nn.Linear(args.v_network_hidden_dim, 1)

        if args.use_orthogonal_init:
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        V = self.fc2(x)
        return V
        
            

class QMIX_Net(nn.Module):
    def __init__(self, args, input_dim):
        super(QMIX_Net, self).__init__()
        self.N = args.n_agents
        self.input_dim = input_dim
        self.qmix_hidden_dim = args.q_mix_hidden_dim
        self.hyper_hidden_dim = args.hyper_hidden_dim
        self.w1_pre_activation = None
        self.b1_pre_activation = None
        self.w2_pre_activation = None
        self.b2_pre_activation = None
        """
        hyper_layers_num: number of layers in hyper network

        """

        # self.hyper_w11 = nn.Linear(self.input_dim, self.N * self.qmix_hidden_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim)
        )
        # self.hyper_w12 = nn.Linear(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim)
        # self.hyper_w12 = nn.GRU(self.hyper_hidden_dim, self.N * self.qmix_hidden_dim, batch_first=True)
        
        # self.hyper_w21 = nn.Linear(self.input_dim, self.qmix_hidden_dim * 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.input_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1)
        )
        # self.hyper_w22 = nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim * 1)
        # self.hyper_w22 = nn.GRU(self.hyper_hidden_dim, self.qmix_hidden_dim * 1, batch_first=True)
        
        # self.hyper_b11 = nn.Linear(self.input_dim, self.qmix_hidden_dim)
        self.hyper_b1 = nn.Sequential(
            nn.Linear(self.input_dim, self.qmix_hidden_dim))
        # self.hyper_b12 = nn.Linear(self.hyper_hidden_dim, self.qmix_hidden_dim)
        # self.hyper_b12 = nn.GRU(self.hyper_hidden_dim, self.qmix_hidden_dim, batch_first=True)
        
        # self.hyper_b21 = nn.Linear(self.input_dim, 1)
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.input_dim, self.hyper_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hyper_hidden_dim, 1)
        )
        # self.hyper_b22 = nn.Linear(self.hyper_hidden_dim, 1)
        # self.hyper_b22 = nn.GRU(self.qmix_hidden_dim, 1, batch_first=True)

        if args.use_orthogonal_init:
            orthogonal_init(self.hyper_w1)
            orthogonal_init(self.hyper_w2)
            orthogonal_init(self.hyper_b1)
            orthogonal_init(self.hyper_b2)
            # orthogonal_init(self.hyper_b22)
    
    def compute_weights_and_biases(self, hyper_input):
        hyper_w1_output = self.hyper_w1(hyper_input)
        hyper_b1_output = self.hyper_b1(hyper_input)
        hyper_w2_output = self.hyper_w2(hyper_input)
        hyper_b2_output = self.hyper_b2(hyper_input)
        
        self.w1 = torch.abs(hyper_w1_output).reshape(-1, self.N, self.qmix_hidden_dim)
        self.b1 = hyper_b1_output.reshape(-1, 1, self.qmix_hidden_dim)
        self.w2 = torch.abs(hyper_w2_output).reshape(-1, self.qmix_hidden_dim, 1)
        self.b2 = hyper_b2_output.reshape(-1, 1, 1)
        
        


