from pandas.core.arrays import string_
import torch 
import numpy as np 
import pandas as pd 
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

pd.options.display.max_colwidth = 200

class LabelledDataset(Dataset):
    def __init__(self ,datapath , sentence_encoder = None):
        self.encoder = sentence_encoder
        self.df = pd.read_csv(datapath)
        encoder = LabelEncoder()
        encoder.fit(self.df.category)
        
        self.df['category'] = encoder.transform(self.df['category'])    
        self.data = list((self.df.groupby(['category']).agg({'text': '<::>'.join })).to_dict().values())[0]

        
        self.data = {k:v.split('<::>') for k,v in self.data.items()}
        
        
        

    def __len__(self):
        pass 

    def __getitem__(self , idx, label):
        pass

    def process_df(self ,datapath):
        df = pd.read_csv(datapath)
        encoder = LabelEncoder()
        encoder.fit(df.category)
        self.df['category'] = encoder.transform(df['category'])
        #print(df)

test = LabelledDataset('clinc150/clinc_train.csv')