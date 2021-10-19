from numpy.lib.function_base import sinc
from torch.nn.modules.activation import ReLU
from torch.nn.modules.loss import NLLLoss
from torch.utils.data import DataLoader
from utilities import *
import torch
from random import randint
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

hyper_parameters = {
            'samples_per_batch' : 200,
            'b' : 60,
            'num_samples':2000,
            'lr' : 0.0001
}


class MLPwSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
                    nn.Linear(150,100),
                    nn.ReLU(),
                    nn.Linear(100,100),
                    nn.ReLU(),
                    nn.Linear(100 , 150)
        )

    def forward(self , x):
        return self.mlp(x)


    
samples_per_batch = hyper_parameters['samples_per_batch']


sentence_encoder = 'all-distilroberta-v1'
# sentence_encoder = None
test = LabelledDataset('./clinc150/clinc_full.csv' , sentence_encoder)

sampler = UniformSampler(test.data ,b=hyper_parameters['b'], num_samples=hyper_parameters['num_samples'])

model = MLPwSoftmax()

model = model.cuda()
criterion = CrossEntropyLoss()
optim = Adam(model.parameters() , lr=hyper_parameters['lr'])




train_loader = DataLoader(test , sampler=sampler ,collate_fn = collate_fn )

for n,x in enumerate(train_loader): 
    for epoch in range(samples_per_batch):
        intent,example = randint(a = 0,b= x.shape[0] -1) , randint(a = 0 , b = x.shape[1] - 1) 
        chosen_sample = x[intent,example]
      #  print(intent,example)

        similarity_matrix = torch.tensordot(x , chosen_sample , dims = ([2],[0]))
        
        s_reduced = reduction_function(similarity_matrix ,intent,example, type = 'minmax')
       # print(s_reduced)
        optim.zero_grad()

        outputs = model.forward(s_reduced)

        #label = to_onehot(intent).reshape(1,150).cuda()
        #print(outputs.shape , torch.tensor(intent).reshape(1,1))
        loss = criterion(outputs.reshape(1,150), torch.tensor(intent).reshape(1).cuda())
        loss.backward()
        writer.add_scalar("Loss" , loss,n*hyper_parameters['num_samples'] + epoch)

        optim.step()

        print(n*hyper_parameters['num_samples'] + epoch)
        #print(loss)

        # gold_label = to_onehot(intent)



        
 


