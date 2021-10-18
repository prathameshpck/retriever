import torch

def collate_fn(data):
    """
        Returns obtained data in a matrix form with shape [intents , num_samples_per , embedding size]

        Most commonly it is [150,B,768]
    """

    
    return torch.stack([v for v in data[0].values()])
    