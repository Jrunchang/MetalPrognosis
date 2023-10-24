from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchsummary import summary
import torch
import math
import torch
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.parameter import Parameter

class ProDataset(Dataset):
    def __init__(self, dataframe):
        self.sites = dataframe['ID'].values
        self.sequences = dataframe['sequence'].values
        self.labels = dataframe['label'].values
        self.embedds = dataframe['embedd'].values

    def __getitem__(self, index):
        site_name = self.sites[index]
        sequence = self.sequences[index]
        embedd = self.embedds[index]   
        label = np.array(self.labels[index])

        return site_name, sequence, embedd,label 

    def __len__(self):
        return len(self.labels)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(32 * 3 * 192, 2)  
        self.dropout = nn.Dropout(p=0.4)
        #32 * 3 * 192,256,448
    def forward(self, x):
        # input size [batch_size, 1, 15, 1280]
        x = x.unsqueeze(1) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        
   
        x = self.fc(x)
        x = self.dropout(x)
        return x


class ConvNetWithSelfAttention(nn.Module):
    def __init__(self):
        super(ConvNetWithSelfAttention, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(8)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.self_attention = nn.MultiheadAttention(embed_dim=16 * 3 * 192, num_heads=1)
        
        self.fc = nn.Linear(16 * 3 * 192, 2)  # 32 * 3 * 320，
        self.dropout = nn.Dropout(p=0.4)
        
    def forward(self, x):
        x = x.unsqueeze(1) 
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)
        
        x, _ = self.self_attention(x, x, x)
        
        x = self.fc(x)
        x = self.dropout(x)
        return x



