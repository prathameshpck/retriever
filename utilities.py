from typing import final
import torch
from os import X_OK
from torch.utils.data.sampler import Sampler
import torch
from pandas.core.arrays import string_
import pandas as pd
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer


def collate_fn(data):
    """
        Returns obtained data in a matrix form with shape [intents , num_samples_per , embedding size]

        Most commonly it is [150,B,768]
    """

    
    return torch.stack([v for v in data[0].values()])

class ClassificationDataset(Dataset):
    def __init__(self ,path_to_df , sentence_encoder ) -> None:
        super().__init__()
        self.df = pd.read_csv(path_to_df)
        self.model = SentenceTransformer(sentence_encoder)

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        return self.model.encode(self.df.iloc[index]['text']) , self.df.iloc[index]['category']


class LabelledDataset(Dataset):
    def __init__(self ,datapath , sentence_encoder = None):             
        self.data = self.df_to_dict(datapath)
        if sentence_encoder:
            self.data = self.encode(sentence_encoder)
        print("Dataset loaded successfully")
       

    def __len__(self):
        return sum(map(lambda x: len(self.data[x]), self.data))

    def __getitem__(self , x):
        
        batch = {k:self.data[k][v] for k,v in x[0].items()}
        return batch
        

    def encode(self , encoder ):
        model = SentenceTransformer(encoder)
        return {k:model.encode(v , convert_to_tensor=True) for k,v in self.data.items()}

    
    def df_to_dict(self , path_to_df ):
        self.df = pd.read_csv(path_to_df)
        self.encoder = LabelEncoder()
        self.encoder.fit(self.df.category)
        self.df['category'] = self.encoder.transform(self.df['category'])    
        self.data = list((self.df.groupby(['category']).agg({'text': '<::>'.join })).to_dict().values())[0]
      
        return {k:v.split('<::>') for k,v in self.data.items()}



class UniformSampler(Sampler):
    """
        Sampler to randomly, uniformly sample from the labelled 

    
    """

    def __init__(self, data_source , b = 5 , num_samples = 10):
        self.data = data_source
        self.b = b
        self.num_samples= num_samples

    def __iter__(self):
        
        # result = [] 
        # for key in self.data.keys():
        #     indices = torch.randperm(len(self.data[key]))[:self.b]
        #     result.append(self.data[key][indices])
           
        # return iter(torch.stack(result))

        content = []
        for i in range(self.num_samples):
                sample = {}
                for key in self.data.keys():
                    
                    if len(self.data[key]) < self.b:
                        indices = list(range(len(self.data[key]))) 
                        extra_samples =  torch.randint(0 , len(self.data[key]),(self.b-len(indices),) )
                        final_indices = indices + extra_samples.tolist()
                        #print(len(final_indices))
                        #print(len(final_indices) , len(indices) , len(extra_samples))

                    else:

                        final_indices = torch.randperm(len(self.data[key]))[:self.b]
                    #print(indices)
                        final_indices = final_indices.tolist()
                    sample[key] = final_indices
                    
                content.append(sample)
             
                
                
                yield (content)
                
        

    def __len__(self):
        pass



def reduction_function(similarity_matrix , intent , example , type = 'mean'):
    if type == 'mean':
        return torch.mean(similarity_matrix , dim = 1)
    elif type == 'max':
        similarity_matrix[intent,example] = -2
        return torch.max(similarity_matrix ,  dim = 1).values
    elif type == 'minmax':
        similarity_matrix[intent] = -similarity_matrix[intent]
        return torch.max(similarity_matrix , dim = 1).values
    else: 
        raise Exception("Choose among mean , max , minmax")

def to_onehot(index , num_classes = 150):
    vector = torch.zeros(num_classes , dtype=torch.long)
    vector[index] = 1
    return vector