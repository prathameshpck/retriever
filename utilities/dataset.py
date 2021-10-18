from pandas.core.arrays import string_
import numpy as np 
import pandas as pd
import torch
from torch.utils import data 
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

pd.options.display.max_colwidth = 200
sentence_encoder = 'all-distilroberta-v1'

class LabelledDataset(Dataset):
    def __init__(self ,datapath , sentence_encoder = None):             
        self.data = self.df_to_dict(datapath)
        if sentence_encoder:
            self.data = self.encoder(sentence_encoder)
        #print((self.data[0]))
       

    def __len__(self):
        return sum(map(lambda x: len(self.data[x]), self.data))

    def __getitem__(self , x):
        
        batch = {k:self.data[k][v] for k,v in x[0].items()}
        return batch
        

    def encoder(self , encoder ):
        model = SentenceTransformer(encoder)
        return {k:model.encode(v , convert_to_tensor=True) for k,v in self.data.items()}

    
    def df_to_dict(self , path_to_df ):
        self.df = pd.read_csv(path_to_df)
        encoder = LabelEncoder()
        encoder.fit(self.df.category)
        self.df['category'] = encoder.transform(self.df['category'])    
        self.data = list((self.df.groupby(['category']).agg({'text': '<::>'.join })).to_dict().values())[0]
      
        return {k:v.split('<::>') for k,v in self.data.items()}

#test = LabelledDataset('./clinc150/clinc_full.csv' , sentence_encoder)