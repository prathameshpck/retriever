from os import X_OK
from torch.utils.data.sampler import Sampler
import torch

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
                    indices = torch.randperm(len(self.data[key]))[:self.b]
                    #print(indices)
                    sample[key] = indices.tolist()
                content.append(sample)
                

                yield (content)
                
        

    def __len__(self):
        pass
