from numpy.lib.function_base import sinc
from torch.nn.modules.activation import ReLU
from torch.nn.modules.loss import NLLLoss
from torch.utils.data import DataLoader
from utilities import *
import torch
import csv
import time
import numpy as np
from random import randint
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import classification_report,confusion_matrix
from tqdm import tqdm 
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# Num intents legend 
# clinc = 150 
# banking = 77 
# hwu = 63



dataset = 'banking'


if dataset == 'hwu':
  num_intents = 63
elif dataset == 'banking':
  num_intents = 77
elif dataset == 'clinc':
  num_intents = 150

hyper_parameters = {
            'samples_per_batch' : 2000,
            'b' : 150,
            'num_samples':1,
            'lr' : 0.000005,
            'reduction_function':'max',
            'batch_size':64, 
            'architecture': [num_intents,100,100,num_intents]
}



class MLPwSoftmax(nn.Module):
    def __init__(self, network_architecure):
        super().__init__()
        # self.input_layer = nn.Linear(num_intents,100)
        # self.mlp = nn.Sequential(
        #             nn.ReLU(),
        #             # nn.Linear(100,100),
        #             # nn.ReLU(),
        #             nn.Linear(100,100),
        #             nn.ReLU()

        # )
        # self.output_layer = nn.Linear(100 , num_intents)
        arch = []
        for layer in range(len(network_architecure) - 1):
          arch.append(nn.Linear(network_architecure[layer] , network_architecure[layer+1]))
          arch.append(nn.ReLU())

        self.mlp = nn.Sequential(*arch[:-1])
          



    def forward(self , x):   
        return self.mlp(x)


    
samples_per_batch = hyper_parameters['samples_per_batch']

model = MLPwSoftmax(hyper_parameters['architecture'])


sentence_encoder = 'all-distilroberta-v1'
# sentence_encoder = None
train_dataset = LabelledDataset('./'+dataset+'_train.csv' , sentence_encoder)
test_dataset = ClassificationDataset('./'+dataset+'_ibm.csv', sentence_encoder)

sampler = UniformSampler(train_dataset.data ,b=hyper_parameters['b'], num_samples=hyper_parameters['num_samples'])



model = model.cuda()
criterion = CrossEntropyLoss()
optim = Adam(model.parameters() , lr=hyper_parameters['lr'])




train_loader = DataLoader(train_dataset , sampler=sampler ,collate_fn = collate_fn,batch_size=hyper_parameters['batch_size'] )
test_loader = DataLoader(test_dataset , batch_size=1)

start_time = time.time()

for n,x in tqdm(enumerate(train_loader)): 
    retrieval_index = x
    x = x.cuda()
    for epoch in range(samples_per_batch):
        intent,example = randint(a = 0,b= x.shape[0] -1) , randint(a = 0 , b = x.shape[1] - 1) 
        chosen_sample = x[intent,example]
      #  print(intent,example)

        similarity_matrix = torch.tensordot(x , chosen_sample , dims = ([2],[0]))
        #print(similarity_matrix)
        s_reduced = reduction_function(similarity_matrix ,intent,example, type = hyper_parameters['reduction_function'])
       # print(s_reduced)
        optim.zero_grad()
        #(s_reduced.shape)
        outputs = model.forward(s_reduced)

        #label = to_onehot(intent).reshape(1,150).cuda()
        #print(outputs.shape , torch.tensor(intent).reshape(1,1))
        loss = criterion(outputs.reshape(1,num_intents), torch.tensor(intent).reshape(1).cuda())
        loss.backward()
        writer.add_scalar("Loss" , loss,n*hyper_parameters['num_samples'] + epoch)

        optim.step()

        #print(n*hyper_parameters['num_samples'] + epoch)
        #print(loss)

        # gold_label = to_onehot(intent)

end_time = time.time()
elapsed_time = end_time - start_time


# retrieval_sampler = UniformSampler(train_dataset.data , b = 150 , num_samples = 1)
# retrieval_dataloader = DataLoader(train_dataset , sampler = retrieval_sampler , collate_fn = collate_fn)
# print(retrieval_sampler.__iter__())

# for r in retrieval_dataloader:
#   retrieval_index = r

total = 0
correct = 0
predicted = []
actual = []

for data in test_loader:
  x, y = data
 
  x = x.cuda()
  y = y[0]
  actual.append(y)
  
  
  similarity_matrix = torch.tensordot(retrieval_index , x , dims = ([2] , [1]))
  s_reduced = reduction_function(similarity_matrix ,intent,example, type = 'max')
  s_reduced = s_reduced.reshape(num_intents)
  
  output = model.forward(s_reduced)

  output = nn.functional.softmax(output)
 
  pred = torch.argmax(output)
 

  pred = train_dataset.encoder.inverse_transform([pred.item()])[0]
  predicted.append(pred)
  if pred == y:
    correct +=1

  else:
    print(pred , y)

  total +=1

print(accuracy :=correct/total)
print(total)


print(classification_report(predicted , actual))
print(confusion_matrix(actual , predicted , normalize='true').diagonal())

torch.save(model.mlp , "model.pt")
 

with open('logsv2.csv' , 'a') as csvfile:
  csvwriter = csv.writer(csvfile)
  csvwriter.writerow([dataset+'train' , num_intents , elapsed_time , hyper_parameters['num_samples'] , hyper_parameters['b'] , hyper_parameters['lr'] , hyper_parameters['samples_per_batch'] ,hyper_parameters['architecture'] , hyper_parameters['batch_size'] , hyper_parameters['reduction_function'] , accuracy])
