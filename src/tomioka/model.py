import torch
import torch.nn as nn
import torch.nn.functional as F

class Classifier(nn.Module):
    def __init__(self, voc_n, pad_index, hid_n, emb_size, dropout=0.5, class_n=6, is_bidirection=False):
        super().__init__()
        self.hid_n = hid_n
        self.emb_size = emb_size
        self.dropout = dropout
        self.class_n = class_n
        self.is_bidirection = is_bidirection

        self.embedding = nn.Embedding(voc_n, emb_size, padding_idx=pad_index)
        self.gru = nn.GRU(emb_size, hid_n, batch_first=True, dropout=dropout, bidirectional=is_bidirection)
        self.fc1 = nn.Linear(hid_n * 2, 500)
        self.fc2 = nn.Linear(500, 300)
        self.fc3 = nn.Linear(300, 200)
        self.final_fc = nn.Linear(200, class_n)
        self.softmax = nn.LogSoftmax()
    
    def forward(self, inp, hid):
        inp = self.embedding(inp)
        _, final_hid = self.gru(inp, hid)
        if self.is_bidirection:
            final_hid = torch.cat((final_hid[-2,:,:], final_hid[-1,:,:]), dim=1)
        else:
            final_hid = torch.cat((final_hid[-1,:,:], final_hid[-1,:,:]), dim=1)
        
        out = F.relu(self.fc1(final_hid))
        out = F.relu(self.fc2(out))
        out = F.relu(self.fc3(out))

        out = self.final_fc(out)

        return out
    
    def init_hidden(self, inp):
        if self.is_bidirection:
            return torch.zeros(1 * 2, inp.size(0), self.hid_n)
        else:
            return torch.zeros(1 * 1, inp.size(0), self.hid_n)