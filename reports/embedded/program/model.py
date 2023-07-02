import torch
import torch.nn as nn


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.num_layers = 1
        self.hidden_size = 128
        self.hidden = torch.zeros(self.num_layers, 64, self.hidden_size)
        self.rnn = nn.GRU(33, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(128, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        self.ip_embedding = nn.Embedding(256, 1)
        self.port_embedding = nn.Embedding(70000, 4)

        self.sigmoid = nn.Softmax(dim=1)

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, 64, self.hidden_size)
        self.hidden = hidden
        return hidden

    def forward(self, x, ip=None, port=None):
        # ip = x[1]
        # port = x[2]
        # x = x[0]
        ip = self.ip_embedding(ip).flatten(1)
        port = self.port_embedding(port).flatten(1)
        x = torch.cat((x, ip, port), dim=1).unsqueeze(1)
        out, self.hidden = self.rnn(x, self.hidden)
        out = self.linear(out)
        self.hidden = self.hidden.detach()
        return self.sigmoid(out.squeeze(1))
