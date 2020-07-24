import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class NNPred(nn.Module):

    def __init__(self, args, dropout=0.5):
        super(NNPred, self).__init__()
        
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

        self.batch_size = args.batch_size
        self.teacher_forcing_ratio = 0.5#if it increases, more label is used for training
        
        self.using_cuda = args.cuda
        self.device = args.device
        self.train_flag = args.train


        # self.hidden_size = hidden_size
        hidden_size = 256
        self.in_length = args.obs_length
        self.out_length = args.pred_length
        self.num_network_output = 2#mx,my,sx,sy,no correlation
        # self.num_layers = 2

        self.in2gru = nn.Linear(2, hidden_size)#Embedding
        self.bn_gru = torch.nn.BatchNorm1d(hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers = 2, batch_first=True)
        self.gruCell0 = nn.GRUCell(hidden_size, hidden_size)
        self.gruCell1 = nn.GRUCell(hidden_size, hidden_size)
        # self.bilstm = nn.LSTM(hidden_size, hidden_size//2,num_layers=self.num_layers, batch_first=True,dropout = dropout)

        self.fc0 = nn.Linear(hidden_size,hidden_size//2)
        self.bn_lin0 = torch.nn.BatchNorm1d(hidden_size//2)
        self.fc1 = nn.Linear(hidden_size//2,self.num_network_output)

        self.leaky_relu = nn.LeakyReLU()
        self.elu = torch.nn.ELU()
        self.tanh = torch.nn.Tanh()
        # self.in2out = nn.Linear(input_size, 64)

        
    def forward(self, input, label):
        
        last_inputEmbedding, last_hidden = self.Encoder(input)
        output = self.Decoder(last_inputEmbedding, last_hidden, label)
        # print("output.shape: ", output.shape)
        return output

    def Encoder(self, input):
        input = self.in2gru(input)
        # print("encode1: ", input.shape)
        input = input.permute(0, 2, 1)# Becuase it is a sequence
        inputEmbedding = self.bn_gru(input).permute(0, 2, 1)#[batch_size, input_sequence, hidden_size(feature)]
    
        inputEmbedding = self.elu(inputEmbedding)
        # print("encode2: ", inputEmbedding.shape)
        _, hidden_n = self.gru(inputEmbedding)#[batch_size, input_sequence, hidden_size(feature)], [num_layers, batch_size, last_hidden(feature)]
        last_inputEmbedding = inputEmbedding[:,-1,:]#[batch_size, last_input_feature]
        # print("encode_last: ", last_inputEmbedding.shape)
        # print("last_hidden: ", hidden_n)
        
        return last_inputEmbedding, hidden_n

    def Decoder(self, input, hidden, label):#2layer decoder
        
        output = torch.zeros((self.batch_size, self.out_length, self.num_network_output))
        if self.cuda:
            output = output.cuda(self.device)
        hidden0 = hidden[0]
        hidden1 = hidden[1]
        for i in range(self.out_length):
            if i != 0:
                if self.train_flag and (random.random() < self.teacher_forcing_ratio):#only training
                    #only for self.num_network_ouput >=2
                    # cat_label = torch.cat((label[:,i-1,:], pred_input[:,2:]), 1)
                    # input = self.bn_gru(self.in2gru(cat_label))
                    input = self.bn_gru(self.in2gru(label[:,i-1,:]))
                    input = self.elu(input)#[batch_size, output_sequence, hidden_size(feature)]
                else:
                    input = self.bn_gru(self.in2gru(pred_input[:,:2]))
                    input = self.elu(input)
            # print("decoder input: ", input.shape)
            # print("hidden0: ", hidden0.shape)
            # print("hidden1: ", hidden1.shape)
            hidden0 = self.gruCell0(input, hidden0)
            hidden1 = self.gruCell1(hidden0, hidden1)
            pred_input = self.FClayers(hidden1)#Get predicted
        
            output[:,i,:] = pred_input
        # print(output)
        # quit()
        return output
        
    
    def FClayers(self, input):
        input = self.bn_lin0(self.fc0(input))
        input = self.elu(input)
        pred_input = self.fc1(input)#Get predicted

        return pred_input
