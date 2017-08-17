"""Implementation of batch-normalized LSTM."""
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional, init
import torch.utils.data as data
import pickle
import numpy as np
import argparse
import torch
import time
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.nn import functional, init


class RHNCell(nn.Module):


    def __init__(self, input_size, hidden_size, use_bias=True):


        super(RHNCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.use_bias = use_bias
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        if use_bias:
            self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize parameters following the way proposed in the paper.
        """

        #nn.init.orthogonal(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data.set_(weight_hh_data)
        # The bias is just set to zero vectors.
        #if self.use_bias:
        #    init.constant(self.bias.data, val=0)

    def forward(self, input_, h_0):
        """
        Args:
            input_: A (batch, input_size) tensor containing input
                features.
            hx: A tuple (h_0, c_0), which contains the initial hidden
                and cell state, where the size of both states is
                (batch, hidden_size).

        Returns:
            h_1, c_1: Tensors containing the next hidden and cell state.
        """

        batch_size = h_0.size(0)
        bias_batch = (self.bias.unsqueeze(0)
                      .expand(batch_size, *self.bias.size()))

        wh_b = torch.addmm(bias_batch, h_0, self.weight_hh)
        wi = torch.mm(input_, self.weight_ih)

        h, t, c = torch.split(wh_b + wi,
                                 split_size=self.hidden_size, dim=1)

        h_t = torch.tanh(h)
        t = torch.sigmoid(t)
        c = torch.sigmoid(t)

        s_t = h_t * t + h_0 * c

        return s_t

    def __repr__(self):
        s = '{name}({input_size}, {hidden_size})'
        return s.format(name=self.__class__.__name__, **self.__dict__)


class RHN(nn.Module):

    """A module that runs multiple steps of LSTM."""

    def __init__(self, input_size, hidden_size, num_layers=1,
                 use_bias=True, batch_first=False, dropout=0,**kwargs):
        super(RHN, self).__init__()
        self.cell_class = RHNCell
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_bias = use_bias
        self.batch_first = batch_first
        self.dropout = dropout

        self.cells = []
        for layer in range(num_layers):
            layer_input_size = input_size if layer == 0 else hidden_size
            cell = self.cell_class(input_size=layer_input_size,
                              hidden_size=hidden_size,
                              **kwargs)
            self.cells.append(cell)
            setattr(self, 'cell_{}'.format(layer), cell)
        self.dropout_layer = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        for cell in self.cells:
            cell.reset_parameters()

    @staticmethod
    def _forward_rnn(cell, input_, length, hx):
        max_time = input_.size(0)
        output = []

        for time in range(max_time):

            h_next = cell(input_=input_[time], h_0=hx)

            mask = (time < length).float().unsqueeze(1).expand_as(h_next)
            
            h_next = h_next*mask + hx*(1 - mask)

            output.append(h_next)
            hx = h_next

        output = torch.stack(output, 0)
        return output, hx

    def forward(self, input_, length=None, h_0=None):

        if self.batch_first:
            input_ = input_.transpose(0, 1)
        max_time, batch_size, _ = input_.size()
        if length is None:
            length = Variable(torch.LongTensor([max_time] * batch_size))
            if input_.is_cuda:
                length = length.cuda()
        if h_0 is None:
            h_0 = Variable(input_.data.new(batch_size, self.hidden_size).zero_())

        h_n = []
        layer_output = None
        for layer in range(self.num_layers):
            layer_output, layer_h_n = RHN._forward_rnn(
                cell=self.cells[layer], input_=input_, length=length, hx=h_0)
            input_ = self.dropout_layer(layer_output)
            h_n.append(layer_h_n)
        output = layer_output
        h_n = torch.stack(h_n, 0)
        return output, h_n


if __name__ =='__main__':

    X = Variable(torch.randn(32,10,50))

    rhn = RHN( 50,16,batch_first=True)

    print( rhn(X)[1].size() )

