import functools
import sys
import numpy as np
import pandas as pd
import random
import re
import matplotlib.pyplot as plt

import nltk
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from nltk.corpus import stopwords
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset

from utils import *
from utils import tokenize

from typing import Tuple, Dict, List

def init_weights(m):
    if isinstance(m, nn.Embedding):
        nn.init.xavier_normal_(m.weight)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM) or isinstance(m, nn.GRU):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.zeros_(param)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
    return



class HyperParams:
    def __init__(self):
        # Constance hyperparameters. They have been tested and don't need to be tuned.
        self.PAD_INDEX = 0
        self.UNK_INDEX = 1
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        self.STOP_WORDS = set(stopwords.words('english'))
        self.MAX_LENGTH = 256
        self.BATCH_SIZE = 96
        self.EMBEDDING_DIM = 1
        self.HIDDEN_DIM = 100
        self.OUTPUT_DIM = 2
        self.N_LAYERS = 1
        self.DROPOUT_RATE = 0.0
        self.LR = 0.01
        self.N_EPOCHS = 5
        self.WD = 0
        self.OPTIM = "sgd"
        self.BIDIRECTIONAL = False
        self.SEED = 5


class ConstantWithWarmup(torch.optim.lr_scheduler._LRScheduler):
    def __init__(
        self,
        optimizer,
        num_warmup_steps: int,
    ):
        self.num_warmup_steps = num_warmup_steps
        super().__init__(optimizer)

    def get_lr(self):
        if self._step_count <= self.num_warmup_steps:
            # warmup
            scale = 1.0 - (self.num_warmup_steps - self._step_count) / self.num_warmup_steps
            lr = [base_lr * scale for base_lr in self.base_lrs]
            self.last_lr = lr
        else:
            lr = self.base_lrs
        return lr


class IMDB(Dataset):
    def __init__(self, x, y, vocab, max_length=256) -> None:
        """
        :param x: list of reviews
        :param y: list of labels
        :param vocab: vocabulary dictionary {word:index}.
        :param max_length: the maximum sequence length.
        """
        self.x = x
        self.y = y
        self.vocab = vocab
        self.max_length = max_length

    def __getitem__(self, idx: int) -> Dict:
        """
        Return the tokenized review and label by the given index.
        :param idx: index of the sample.
        :return: a dictionary containing three keys: 'ids', 'length', 
        'label' which represent the list of token ids, 
        the length of the sequence, the binary label.
        """
        # Add your code here.
        pn_dict = {'positive': 1, 'negative': 0}
        
        review, binary_label = tokenize(vocab=self.vocab, example=self.x[idx]), pn_dict[self.y[idx]]  
        # print(len(review), review[0:2])
        if len(review) > self.max_length:
            return {'ids': review[0:self.max_length], 'length': self.max_length, 'label': binary_label}
        else:
            return {'ids': review, 'length': len(review), 'label': binary_label}
        
    def __len__(self) -> int:
        return len(self.x)


class LSTM(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        n_layers: int, 
        dropout_rate: float, 
        pad_index: int,
        bidirectional: bool,
        **kwargs):
        """
        Create a LSTM model for classification.
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of embeddings
        :param hidden_dim: dimension of hidden features
        :param output_dim: dimension of the output layer which equals to the number of labels.
        :param n_layers: number of layers.
        :param dropout_rate: dropout rate.
        :param pad_index: index of the padding token.we
        """
        super().__init__()
        # Add your code here. Initializing each layer by the given arguments.
        self.pad_index = pad_index
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, padding_idx=pad_index, 
                                      max_norm=None, norm_type=2.0, scale_grad_by_freq=False, 
                                      sparse=False, _weight=None, _freeze=False, 
                                      device=None, dtype=None)
        lstm_dropout = None
        if n_layers > 1:
            lstm_dropout = dropout_rate
        else:
            lstm_dropout = 0.0
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, 
                            num_layers=n_layers, bias=True, batch_first=True, 
                            dropout=lstm_dropout, bidirectional=bidirectional, proj_size=0, 
                            device=None, dtype=None)
        # dropout=0.0 in self.lstm since n_layers is only 1 and the last lstm layer 
        # doesn't have any dropout
        # batch_first=True since ids is batch_size * sequence_len
        
        self.dropout = nn.Dropout(dropout_rate)
        
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim, 
                            bias=True, device=None, dtype=None)
        if bidirectional: # bidirectional: 2*hidden_dim is needed 
            # since lstm output is double the size
            self.fc = nn.Linear(in_features=2*hidden_dim, out_features=output_dim, 
                                bias=True, device=None, dtype=None)
        # Weight initialization. DO NOT CHANGE!
        if "weight_init_fn" not in kwargs:
            self.apply(init_weights)
        else:
            self.apply(kwargs["weight_init_fn"])

    def forward(self, ids:torch.Tensor, length:torch.Tensor):
        """
        Feed the given token ids to the model.
        :param ids: [batch size, seq len] batch of token ids.
        :param length: [batch size] batch of length of the token ids.
        :return: prediction of size [batch size, output dim].
        """
        # note: using batch_first=True
        # do we need to pad? the dataloader seems to be pretty good at padding
        # NO NEED to use nn.utils.rnn.pad_sequence since collate_fn already pads everything
        # and each batch has already been padded.
        padded_ids = nn.utils.rnn.pad_sequence(sequences=ids, batch_first=True, 
                                               padding_value=self.pad_index)
        
        # Add your code here.
        out = self.embedding(padded_ids)
        # packs the embeddings (better for computation)
        out = nn.utils.rnn.pack_padded_sequence(out, length, batch_first=True, enforce_sorted=False)
        
        # Pass embedded input through the LSTM and dropout layers
        out, hidden = self.lstm(out)
        # otuput of lstm is packed sequence
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, 
                                                  padding_value=self.pad_index)
        # unpack the packed thing, should result in tensor of shape (batch size, seq len, hidden_size)
        # link to explain the range indexing below: take out every length-1-th element for each batch
        # https://stackoverflow.com/questions/53123009/using-python-range-objects-to-index-into-numpy-arrays
        out = self.dropout(out[range(out.shape[0]), length.int() - 1, :])
        # dropout(tensor of shape [batch_size, hidden_dim]) i.e. apply dropout on 
        # last hidden state of each input in the batch
        
        prediction = self.fc(out)
        # Get the output from the last time step
        return prediction
    

class GRU(nn.Module):
    def __init__(
        self, 
        vocab_size: int, 
        embedding_dim: int, 
        hidden_dim: int, 
        output_dim: int, 
        n_layers: int, 
        dropout_rate: float, 
        pad_index: int,
        bidirectional: bool,
        **kwargs):
        """
        Create a LSTM model for classification.
        :param vocab_size: size of the vocabulary
        :param embedding_dim: dimension of embeddings
        :param hidden_dim: dimension of hidden features
        :param output_dim: dimension of the output layer which equals to the number of labels.
        :param n_layers: number of layers.
        :param dropout_rate: dropout rate.
        :param pad_index: index of the padding token.we
        """
        super().__init__()
        # Add your code here. Initializing each layer by the given arguments.
        self.pad_index = pad_index
        self.embedding = nn.Embedding(num_embeddings=vocab_size, 
                                      embedding_dim=embedding_dim, padding_idx=pad_index, 
                                      max_norm=None, norm_type=2.0, scale_grad_by_freq=False, 
                                      sparse=False, _weight=None, _freeze=False, 
                                      device=None, dtype=None)
        gru_dropout = None
        if n_layers > 1:
            gru_dropout = dropout_rate
        else:
            gru_dropout = 0.0
        
        self.gru = nn.GRU(input_size=embedding_dim, hidden_size=hidden_dim, 
                          num_layers=n_layers, bias=True, batch_first=True, 
                          dropout=gru_dropout, bidirectional=bidirectional, 
                          device=None, dtype=None)
        
        self.fc = nn.Linear(in_features=hidden_dim, out_features=output_dim, 
                            bias=True, device=None, dtype=None)
        if bidirectional:  # hidden_dim*2, same reason as in LSTM
            self.fc = nn.Linear(in_features=hidden_dim*2, out_features=output_dim, 
                                bias=True, device=None, dtype=None)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Weight Initialization. DO NOT CHANGE!
        if "weight_init_fn" not in kwargs:
            self.apply(init_weights)
        else:
            self.apply(kwargs["weight_init_fn"])


    def forward(self, ids:torch.Tensor, length:torch.Tensor):
        """
        Feed the given token ids to the model.
        :param ids: [batch size, seq len] batch of token ids.
        :param length: [batch size] batch of length of the token ids.
        :return: prediction of size [batch size, output dim].
        """
        # Add your code here.
        
        # DO NOT need to use nn.utils.rnn.pad_sequence since collate_fn already pads everything
        # and each batch has already been padded.
        padded_ids = nn.utils.rnn.pad_sequence(sequences=ids, batch_first=True, 
                                               padding_value=self.pad_index)
        out = self.embedding(padded_ids)
        out = nn.utils.rnn.pack_padded_sequence(out, length, batch_first=True, enforce_sorted=False)
        
        out, _ = self.gru(out)  # second output is hidden state
        
        # unpack gru_out since it's packed, results in tensor of shape (batch size, seq len, hidden_size)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, 
                                                  padding_value=self.pad_index)
        
        # Apply dropout to the GRU output
        out = self.dropout(out[range(out.shape[0]), length.int() - 1, :])
        # array indexing: 
        # for each input, take the last non-padded hidden state (which is where length comes in). 
        # ends up with tensor of shape [batch size, hidden_dim]
        
        prediction = self.fc(out)
        return prediction
    
