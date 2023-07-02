import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.num_layers = 1
        self.hidden_size = 128
        self.hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.rnn = nn.GRU(33, self.hidden_size, self.num_layers, batch_first=True)
        self.ip_embedding = nn.Embedding(256, 1)
        self.port_embedding = nn.Embedding(70000, 4)

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, 1, self.hidden_size)
        self.hidden = hidden
        return hidden

    def forward(self, x, ip, port, hidden):
        ip = self.ip_embedding(ip).flatten(1)
        port = self.port_embedding(port).flatten(1)
        x = torch.cat((x, ip, port), dim=1).unsqueeze(1)
        outputs = torch.zeros(64, 128)
        x = x.unsqueeze(1)
        for idx in range(64):
            # out, self.hidden = self.rnn(x[idx], self.hidden)
            out, self.hidden = self.rnn(x[idx], hidden)
            outputs[idx] = out[0, 0]

        self.hidden = self.hidden.detach()
        return outputs, self.hidden


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.num_layers = 1
        self.hidden_size = 128
        self.hidden = torch.zeros(self.num_layers, 64, self.hidden_size)
        self.rnn = nn.GRU(self.hidden_size, self.hidden_size, self.num_layers, batch_first=True)
        self.linear = nn.Sequential(
            nn.Linear(128, 50),
            nn.ReLU(),
            nn.Linear(50, 2)
        )
        self.attn = nn.Linear(self.hidden_size * 2, 64)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.sigmoid = nn.Softmax(dim=1)

    def init_hidden(self):
        hidden = torch.zeros(self.num_layers, 64, self.hidden_size)
        self.hidden = hidden
        return hidden

    def forward(self, encoder_outputs, hidden):
        input = torch.zeros(1, 1, 128)
        output = torch.zeros(64, 128)
        for idx in range(64):
            attn_weights = F.softmax(self.attn(torch.cat((input[0], hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
            # print(attn_applied.shape)
            out = torch.cat((input, attn_applied), 1)
            out = self.attn_combine(out.flatten(1))
            out = F.relu(out)
            out, hidden = self.rnn(out.unsqueeze(0), hidden)
            input = out
            output[idx] = out[0, 0]

        output = self.linear(output)
        return self.sigmoid(output), hidden
