import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim

from data import PAD_TOKEN
from dataloader import get_dataloader
from model import Classifier

def run(dataset_dir, hid_n=128, emb_size=128, batchsize=128, epoch=10, lr=0.01, use_cuda=False, seed=0):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = pickle.load(open(dataset_dir / "dataset.train.token.pkl", 'rb'))
    token2index, words = pickle.load(open(dataset_dir / "vocab.pkl", 'rb'))

    device = torch.device('cuda' if use_cuda else 'cpu')

    voc_num = len(token2index)
    pad_index = token2index[PAD_TOKEN]

    fix_max_len = 50

    train_dataloader = get_dataloader(train_dataset, batchsize, pad_index, fix_max_len=fix_max_len)

    model = Classifier(voc_num, pad_index, hid_n, emb_size, dropout=0)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    loss_func = nn.CrossEntropyLoss()

    model.train()

    for i_epoch in tqdm(range(1, epoch + 1), total=epoch):
        losses = []
        for labels, inputs in train_dataloader:
            labels = labels.to(device)
            inputs = labels.to(device)

            hid = model.init_hidden(inputs)
            hid = hid.to(device)

            out = model(inputs, hid)
            loss = loss_func(out, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        tqdm.write(str(np.mean(losses)))

if __name__ == "__main__":
    run(Path("./datas/datasets"))