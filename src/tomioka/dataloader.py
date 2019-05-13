import torch
import torch.utils.data as data
from torch.utils.data.sampler import RandomSampler


def get_collate_fn(pad_index, fix_max_len=None):

    def _f(batch):
        tokens_list = []
        labels = []

        for ins in batch:
            tokens_list.append(ins["tokens"])
            labels.append(ins["label"])
        
        if not fix_max_len:
            max_len = max(len(tokens) for tokens in tokens_list)
        else:
            max_len = fix_max_len
        
        padded_tokens_list = []
        for tokens in tokens_list:
            if len(tokens) < max_len:
                padded_tokens = tokens + [pad_index] * (max_len - len(tokens))
            else:
                padded_tokens = tokens[:max_len]
            padded_tokens_list.append(padded_tokens)
        
        return (torch.LongTensor(labels), torch.LongTensor(padded_tokens_list))
    
    return _f

def get_dataloader(dataset, batchsize, pad_index, fix_max_len=None, shuffle=True):
    if shuffle:
        dataloader = data.DataLoader(
            dataset,
            batch_size=batchsize,
            sampler=RandomSampler(dataset),
            collate_fn=get_collate_fn(pad_index, fix_max_len)
            )
    else:
        dataloader = data.DataLoader(
            dataset,
            batch_size=batchsize,
            shuffle=False,
            collate_fn=get_collate_fn(pad_index, fix_max_len)
        )
    return dataloader