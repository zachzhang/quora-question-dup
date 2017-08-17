import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
import torch.utils.data as data
import pickle
import numpy as np

from rhn import *

class QuestionModel(nn.Module):


    def __init__(self,embedding, highway, n_dense):

        super( QuestionModel, self).__init__()
        h = 75
        p = .2
        self.highway = highway

        self.embed = nn.Embedding(embedding.size()[0], embedding.size()[1], padding_idx=0 )
        self.embed.weight = nn.Parameter(embedding , requires_grad = False)

        if not self.highway:
            self.rnn = nn.LSTM(embedding.size()[1] , h , batch_first = True, dropout = p )
            print('Using LSTM')
        else:
            self.rnn = RHN(embedding.size()[1], h , batch_first =True ,  dropout = p)
            print('Using RHN')

        self.nlp_dense = nn.Linear(n_dense,200)
       

        self.fc = nn.Linear(2*h + 200, 150 ,bias=False)
        self.output = nn.Linear(150,1)
        self.model = nn.Sequential()
        self.h = h
        self.gpu = False

        self.bn1 = nn.BatchNorm1d(n_dense)
        self.bn2 = nn.BatchNorm1d(2*h + 200)
        self.bn3 = nn.BatchNorm1d(150)

        #self.params = list(self.fc.parameters()) + list(self.rnn.parameters()) + list(self.output.parameters() + list(self.nlp_dense.parameters()) +\

                # list(self.bn1.parameters()) + list(self.bn2.parameters()) + list(self.bn3.parameters())


    def forward(self,q1,q2,features):


        h0 = Variable(torch.zeros(1, q1.size()[0], self.h))
        c0 = Variable(torch.zeros(1, q2.size()[0], self.h))

        if self.gpu:
            h0 = h0.cuda()
            c0 = c0.cuda()

        e_q1 = self.embed(q1)
        e_q2 = self.embed(q2)
        
        if self.highway:

            h_q1 = self.rnn(e_q1)[1].squeeze()
            h_q2 = self.rnn(e_q2)[1].squeeze()


        else:
            h_q1 = self.rnn(e_q1,(h0,c0))[1][0].squeeze()
            h_q2 = self.rnn(e_q2,(h0,c0))[1][0].squeeze()
        
        h_dense = F.dropout(F.relu(self.nlp_dense( self.bn1(features))),.2)

        h =F.dropout( torch.cat( [(h_q1 - h_q2).abs() , h_q1 * h_q2 ] , 1) ,.4)
        h = torch.cat([h,h_dense],1)
        h = self.bn2(h)


        h = F.dropout(F.relu( self.fc(h) ) ,1)
        h = self.bn3(h)

        return self.output(h)

#model = QuestionModel(torch.randn(50,100),True, 10)


#print(model(Variable(torch.ones(16,12)).long() , Variable(torch.ones(16,12)).long() , Variable(torch.ones(16,10))).size())


