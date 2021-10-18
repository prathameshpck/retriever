from torch.utils.data import DataLoader
from utilities.dataset import LabelledDataset
from utilities.sampler import UniformSampler
from utilities.collate import collate_fn
import torch
from random import randint

samples_per_batch = 10


sentence_encoder = 'all-distilroberta-v1'
# sentence_encoder = None
test = LabelledDataset('./clinc150/clinc_train_10.csv' , sentence_encoder)

sampler = UniformSampler(test.data)




train_loader = DataLoader(test , sampler=sampler ,collate_fn = collate_fn )

for x in train_loader: 
    for _ in range(samples_per_batch):
        i,j = randint(a = 0,b= x.shape[0] -1) , randint(a = 0 , b = x.shape[1] - 1) 
        chosen_sample = x[i,j]
        print(i,j)

        similarity_matrix = torch.tensordot(x , chosen_sample , dims = ([2],[0]))
        print(torch.max(similarity_matrix))
        print(similarity_matrix)
        break 
    break
