import torch

def collate_fn(data):
    
    return torch.stack([v for v in data[0].values()])
    