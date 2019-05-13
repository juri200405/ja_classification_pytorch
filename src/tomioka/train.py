import pickle
from pathlib import Path

import numpy as np
import torch

def run(dataset_dir, hid_n=128, emb_size=128, batchsize=128, epoch=10, lr=0.01, use_cuda=False, seed=0):
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = pickle.load(open(dataset_dir / "dataset.train.token.pkl", 'rb'))
    token2index = pickle.load(open(dataset_dir / "vocab.pkl", 'rb'))

    print(train_dataset[0])

if __name__ == "__main__":
    run(Path("./datas/datasets"))