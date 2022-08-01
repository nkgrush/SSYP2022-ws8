from torch import nn
import torch

class RNNCell(nn.Module):
    def __init__(self, input_dim, hid_dim, device='cpu'):
        super().__init__()

        self.state2hid = nn.Linear(input_dim + hid_dim, hid_dim)
        self.state2out = nn.Linear(input_dim + hid_dim, hid_dim)
        self.act = nn.Sigmoid()
        self.hid_dim = hid_dim
        self.device = device

    def forward_once(self, input, hidden):
        concat = torch.cat((input, hidden), 1)
        concat = self.act(concat)
        next_hid = self.state2hid(concat)
        next_out = self.state2out(concat)

        return next_out, next_hid

    def forward(self, input_batch, hidden=None):
        hid_list = []
        out_list = []
        if hidden is None:
            hidden = self.initHidden(input_batch.shape[1])

        for word_batch_input in input_batch:
            next_out, next_hid = self.forward_once(word_batch_input, hidden)
            hid_list.append(next_hid)
            out_list.append(next_out)
            hidden = next_hid

        return torch.stack(out_list), torch.stack(hid_list)

    def initHidden(self, batch_size):
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hid_dim)).to(self.device)


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, classes, input_dim, hid_dim):
        super().__init__()
        hid_dim = 4
        self.emb = nn.Embedding(vocab_size, input_dim)
        self.rnn_cell = RNNCell(input_dim, hid_dim)
        self.fc = nn.Linear(hid_dim, classes)

    def forward(self, x, original_lens):
        x = self.emb(x)

        out, hid = self.rnn_cell(x)
        print(out.shape)
        last_out = out[-1]
        x = self.fc(last_out)
        return x
