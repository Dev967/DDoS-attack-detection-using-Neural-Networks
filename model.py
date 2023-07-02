import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, num_layers, hidden_size, bidirectional, batch_size=64, ip_embedding_size=1,
                 port_embedding_size=4, path=None, name="encoder", desc=None, activation_fn=nn.ReLU(),
                 dropout=0):
        super(Encoder, self).__init__()
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.path = path
        self.name = name
        self.activation = activation_fn

        # RNN/GRU
        # self.rnn = nn.GRU(33, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # LSTM
        self.rnn = nn.LSTM(33, self.hidden_size, self.num_layers, batch_first=True, dropout=dropout)
        self.ip_embedding = nn.Embedding(256, ip_embedding_size)
        self.port_embedding = nn.Embedding(70000, port_embedding_size)

        if self.path:
            desc_file = open(f'{self.path}/{self.name}_desc.txt', 'w')
            desc_file.writelines(f'DESCRIPTION: \n {desc} \n\n')
            desc_file.writelines(str(self))
            desc_file.close()

    def save(self):
        if self.path: torch.save(self.state_dict(), f'{self.path}/{self.name}.pt')

    def load(self):
        if self.path: self.load_state_dict(torch.load(f'{self.path}/{self.name}.pt'))

    def init_hidden(self):
        hidden = torch.zeros(self.bidirectional * self.num_layers, 1, self.hidden_size)
        return hidden

    def init_cell(self):
        cell_state = torch.zeros(self.bidirectional * self.num_layers, 1, self.hidden_size)
        return cell_state

    def forward(self, x, ip, port, hidden, cell=None):
        ip = self.ip_embedding(ip).flatten(1)
        port = self.port_embedding(port).flatten(1)
        x = torch.cat((x, ip, port), dim=1).unsqueeze(1)
        outputs = torch.zeros(self.batch_size, self.hidden_size)
        x = x.unsqueeze(1)

        for idx in range(self.batch_size):
            # RNN/GRU
            # out, hidden = self.rnn(x[idx], hidden)
            # LSTM
            out, (hidden, cell) = self.rnn(x[idx], (hidden, cell))
            out = self.activation(out)
            outputs[idx] = out[0, 0]

        # RNN/GRU
        # hidden.detach_()
        # return outputs, hidden

        # LSTM
        hidden.detach_()
        cell.detach_()
        return outputs, (hidden, cell)


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, bidirectional, batch_size=64, desc=None, name="decoder", path=".",
                 activation_fn=nn.ReLU(), other_activation_fn=nn.Softmax(dim=1), dropout=0):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.batch_size = batch_size
        self.path = path
        self.name = name
        self.activation = activation_fn

        # RNN/GRU
        # self.rnn = nn.GRU(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        # LSTM
        self.rnn = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Sequential(
            nn.Linear(hidden_size, 64),
            self.activation,
            nn.Linear(64, 2),
        )
        self.attn = nn.Linear(hidden_size * 2, batch_size)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)
        self.sigmoid = other_activation_fn

        desc_file = open(f'{path}/{self.name}_desc.txt', 'w')
        desc_file.writelines(f'DESCRIPTION: \n{desc}  \n\n')
        desc_file.writelines(str(self))
        desc_file.close()

    def save(self):
        torch.save(self.state_dict(), f'{self.path}/{self.name}.pt')

    def load(self):
        self.load_state_dict(torch.load(f'{self.path}/{self.name}.pt'))

    def init_hidden(self):
        hidden = torch.zeros(self.bidirectional * self.num_layers, self.batch_size, self.hidden_size)
        return hidden

    def init_cell(self):
        hidden = torch.zeros(self.bidirectional * self.num_layers, self.batch_size, self.hidden_size)
        return hidden

    def forward(self, encoder_outputs, hidden, cell=None):
        decoder_input = torch.zeros(1, 1, self.hidden_size)
        output = torch.zeros(self.batch_size, self.hidden_size)
        for idx in range(self.batch_size):
            attn_weights = F.softmax(self.attn(torch.cat((decoder_input[0], hidden[0]), 1)), dim=1)
            attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))
            out = torch.cat((decoder_input, attn_applied), 1)
            out = self.attn_combine(out.flatten(1))
            out = self.activation(out)
            # RNN/GRU
            # out, hidden = self.rnn(out.unsqueeze(0), hidden)
            # LSTM
            out, (hidden, cell) = self.rnn(out.unsqueeze(0), (hidden, cell))
            decoder_input = out
            output[idx] = out[0, 0]

        # RNN/GRU
        # output = self.linear(output)
        # return self.sigmoid(output), hidden
        # LSTM
        output = self.linear(output)
        return self.sigmoid(output), (hidden, cell)
