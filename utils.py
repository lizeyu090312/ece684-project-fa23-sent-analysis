import functools
import sys
import numpy as np
import pandas as pd
import random
import re
import os
import inspect

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
import torch.nn.functional as F

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


def tokenize(vocab: dict, example: str) -> list:
    """
    Tokenize the give example string into a list of token indices.
    :param vocab: dict, the vocabulary.
    :param example: a string of text.
    :return: a list of token indices.
    """
    # Your code here.
    ex_tok = example.split()
    ex = [w.lower() for w in ex_tok]
    # return [vocab[w] if w in vocab else vocab[ORIG_HPARAMS.UNK_TOKEN] for w in ex]
    return [vocab[w] for w in ex if w in vocab]


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


ORIG_HPARAMS = HyperParams()


def diff_hparams(orig_hp: HyperParams, new_hp: HyperParams) -> str:
    """
    @param orig_hp: original hyperparameters, class HyperParams
    @param new_hp: hyperparameters that have different values
    return string containing attr and vals that are different 
    e.g.
    org_hyperparams = HyperParams(); new_hyperparams = HyperParams()
    new_hyperparams.N_LAYERS = 10; new_hyperparams.BIDIRECTIONAL = True
    diff_hparams(org_hyperparams, new_hyperparams)
    
    return 'BIDIRECTIONAL=True,N_LAYERS=10'
    """
    attr_orig = inspect.getmembers(orig_hp, lambda a:not(inspect.isroutine(a)))
    attr_orig_filtered = [a for a in attr_orig if not(a[0].startswith('__') and a[0].endswith('__'))]
    
    attr_new = inspect.getmembers(new_hp, lambda a:not(inspect.isroutine(a)))
    attr_new_filtered = [a for a in attr_new if not(a[0].startswith('__') and a[0].endswith('__'))]
    diff_hparams_dict = {}
    for idx, (n, val) in enumerate(attr_new_filtered):
        if val != attr_orig_filtered[idx][1]:
            diff_hparams_dict[n] = val
    ret = ""
    for idx, (k, v) in enumerate(diff_hparams_dict.items()):
        if type(v) == int or type(v) == float:
            ret = ret + '%s=%g,' % (k, v)
        if type(v) == bool:
            ret = ret + '%s=%r,' % (k, v)
        if type(v) == str:
            ret = ret + '%s=%s,' % (k, v)
        if idx % 4 == 0:
            ret = ret + '\n'
    return ret[0:-1]


def load_imdb(base_csv:str = './IMDBDataset.csv'):
    """
    Load the IMDB dataset
    :param base_csv: the path of the dataset file.
    :return: train, validation and test set.
    """
    # Add your code here. 
    df = pd.read_csv(base_csv)
    x = list(df['review'])
    y = list(df['sentiment'])
    # 7:3 train/(test&Val) ratio
    x_train, x_testVal, y_train, y_testVal = train_test_split(x, y, test_size=0.3, 
                                                              random_state=1002, 
                                                              shuffle=True)
    # 1:2 val='train'/test ratio
    x_valid, x_test, y_valid, y_test = train_test_split(x_testVal, y_testVal, 
                                                        test_size=float(2/3), 
                                                        random_state=1002, 
                                                        shuffle=True)
    # I'm only to run this function once, so it shouldn't matter across different trainings if I shuffle. 
    # random seed is set, so should be constant across different kernel sessions.
    print(f'shape of train data is {len(x_train)}')
    print(f'shape of test data is {len(x_test)}')
    print(f'shape of valid data is {len(x_valid)}')
    return x_train, x_valid, x_test, y_train, y_valid, y_test


def build_vocab(x_train:list, min_freq: int=5, hparams: HyperParams =None, truncate=False) -> dict:
    """
    build a vocabulary based on the training corpus.
    :param x_train:  List. The training corpus. Each sample in the list is a string of text.
    :param min_freq: Int. The frequency threshold for selecting words.
    :return: dictionary {word:index}
    """
    # Add your code here. Your code should assign corpus with a list of words.
    
    # split x_train into words 
    words_temp = []
    for sent in x_train:
        words_lst = sent.split()
        words_lst_lower = [w.lower() for w in words_lst]
        words_temp.extend(words_lst_lower)
    # no additional string preprocessing

    corpus = Counter([w for w in words_temp if w not in hparams.STOP_WORDS])
    
    # sorting on the basis of most common words
    corpus_ = [word for word, freq in corpus.items() if freq >= min_freq]
    if truncate:
        corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    # creating a dict
    vocab = {word : idx + 2 for idx, word in enumerate(corpus_)}
    vocab[hparams.PAD_TOKEN] = hparams.PAD_INDEX
    vocab[hparams.UNK_TOKEN] = hparams.UNK_INDEX
    return vocab

# x_train, x_valid, x_test, y_train, y_valid, y_test = load_imdb()  
x_train, x_valid, x_test, y_train, y_valid, y_test = load_imdb("./IMDB_synthetic_final.csv")
vocab = build_vocab(x_train, hparams=ORIG_HPARAMS, truncate=False)

def collate(batch, pad_index):
    batch_ids = [torch.LongTensor(i['ids']) for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = torch.Tensor([i['length'] for i in batch])
    batch_label = torch.LongTensor([i['label'] for i in batch])
    batch = {'ids': batch_ids, 'length': batch_length, 'label': batch_label}
    return batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_data_l_of_l():
    def l_of_l(x_list, y_list):
        sents = []
        for x in x_list:
            sents.append([word.lower() for word in x.split()])
        return x_list, np.array(y_list)
    return *l_of_l(x_train, y_train), *l_of_l(x_valid, y_valid), *l_of_l(x_test, y_test)


def load_data(trainloader, testloader):
    """
    Exclusively for Naive Bayes
    """
    x_train, y_train, len_train, x_test, y_test, len_test = [[] for _ in range(6)]
    # for batch in trainloader, desc='training...', file=sys.stdout, disable=True):
    for batch in trainloader:
        x_train.append(batch['ids'])
        len_train.append(batch['length'])
        y_train.append(batch['label'])
    for batch in testloader:
        x_test.append(batch['ids'])
        len_train.append(batch['length'])
        y_test.append(batch['label'])
    return x_train, y_train, len_train, x_test, y_test, len_test


def train(dataloader, model, criterion, optimizer, scheduler, device):
    model.train()
    epoch_losses = []
    epoch_accs = []
    # orig: tqdm.tqdm(dataloader, desc='training...', file=sys.stdout) with import tqdm
    # for batch in tqdm(dataloader, desc='training...', file=sys.stdout, miniters=int(223265/100)):
    for batch in tqdm(dataloader, desc='training...', file=sys.stdout, disable=True):
        # my addition
        ids = batch['ids'].to(device)
        length = batch['length']
        label = batch['label'].to(device)
        prediction = model(ids, length)
        loss = criterion(prediction, label)
        accuracy = get_accuracy(prediction, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_accs.append(accuracy.item())
        scheduler.step()

    return epoch_losses, epoch_accs


def evaluate(dataloader, model, criterion, device):
    model.eval()
    epoch_losses = []
    epoch_accs = []

    with torch.no_grad():
        # orig: tqdm.tqdm(dataloader, desc='training...', file=sys.stdout) with import tqdm
        # for batch in tqdm(dataloader, desc='evaluating...', file=sys.stdout, miniters=int(223265/100)):
        for batch in tqdm(dataloader, desc='evaluating...', file=sys.stdout, disable=True):
            # my addition
            ids = batch['ids'].to(device)
            length = batch['length']
            label = batch['label'].to(device)
            prediction = model(ids, length)
            loss = criterion(prediction, label)
            accuracy = get_accuracy(prediction, label)
            epoch_losses.append(loss.item())
            epoch_accs.append(accuracy.item())

    return epoch_losses, epoch_accs


def get_accuracy(prediction, label):
    batch_size, _ = prediction.shape
    predicted_classes = prediction.argmax(dim=-1)
    correct_predictions = predicted_classes.eq(label).sum()
    accuracy = correct_predictions / batch_size
    return accuracy


collate_fn = collate

def get_dataloaders():
    train_data = IMDB(x_train, y_train, vocab, ORIG_HPARAMS.MAX_LENGTH)
    valid_data = IMDB(x_valid, y_valid, vocab, ORIG_HPARAMS.MAX_LENGTH)
    test_data = IMDB(x_test, y_test, vocab, ORIG_HPARAMS.MAX_LENGTH)

    collate = functools.partial(collate_fn, pad_index=ORIG_HPARAMS.PAD_INDEX)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=ORIG_HPARAMS.BATCH_SIZE, collate_fn=collate, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=ORIG_HPARAMS.BATCH_SIZE, collate_fn=collate)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=ORIG_HPARAMS.BATCH_SIZE, collate_fn=collate)
    return train_dataloader, valid_dataloader, test_dataloader


def train_and_test_model_with_hparams(hparams, model_type="lstm", **kwargs):
    CHECKPOINT_FOLDER = "./saved_model/"
    # computes x_train, x_valid... etc once only for all models
    # Seeding. DO NOT TOUCH! DO NOT TOUCH hparams.SEED!
    # Set the random seeds.
    torch.manual_seed(hparams.SEED)
    random.seed(hparams.SEED)
    np.random.seed(hparams.SEED)

    vocab_size = len(vocab)
    print(f'Length of vocabulary is {vocab_size}')
    # use x_train, x_valid, x_test, y_train, y_valid, y_test as global
    # better for comparing betw models.
    train_data = IMDB(x_train, y_train, vocab, hparams.MAX_LENGTH)
    valid_data = IMDB(x_valid, y_valid, vocab, hparams.MAX_LENGTH)
    test_data = IMDB(x_test, y_test, vocab, hparams.MAX_LENGTH)

    collate = functools.partial(collate_fn, pad_index=hparams.PAD_INDEX)

    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders()
    
    # Model
    if "override_models_with_gru" in kwargs and kwargs["override_models_with_gru"]:
        model = GRU(
            vocab_size, 
            hparams.EMBEDDING_DIM, 
            hparams.HIDDEN_DIM, 
            hparams.OUTPUT_DIM,
            hparams.N_LAYERS,
            hparams.DROPOUT_RATE, 
            hparams.PAD_INDEX,
            hparams.BIDIRECTIONAL,
            **kwargs)
    else:
        model = LSTM(
            vocab_size, 
            hparams.EMBEDDING_DIM, 
            hparams.HIDDEN_DIM, 
            hparams.OUTPUT_DIM,
            hparams.N_LAYERS,
            hparams.DROPOUT_RATE, 
            hparams.PAD_INDEX,
            hparams.BIDIRECTIONAL,
            **kwargs)
    num_params = count_parameters(model)
    print(f'The model has {num_params:,} trainable parameters')


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Optimization. Lab 2 (a)(b) should choose one of them.
    # DO NOT TOUCH optimizer-specific hyperparameters! (e.g., eps, momentum)
    # DO NOT change optimizer implementations!
    if hparams.OPTIM == "sgd":
        optimizer = optim.SGD(
            model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, momentum=.9)        
    elif hparams.OPTIM == "adagrad":
        optimizer = optim.Adagrad(
            model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6)
    elif hparams.OPTIM == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6)
    elif hparams.OPTIM == "rmsprop":
        optimizer = optim.RMSprop(
            model.parameters(), lr=hparams.LR, weight_decay=hparams.WD, eps=1e-6, momentum=.9)
    else:
        raise NotImplementedError("Optimizer not implemented!")

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)

    # Start training
    best_valid_loss = float('inf')
    train_losses = []
    train_accs = []
    valid_losses = []
    valid_accs = []
    
    # Warmup Scheduler. DO NOT TOUCH!
    WARMUP_STEPS = 200
    lr_scheduler = ConstantWithWarmup(optimizer, WARMUP_STEPS)

    for epoch in range(hparams.N_EPOCHS):
        
        # Your code: implement the training process and save the best model.

        train_loss, train_acc = train(dataloader=train_dataloader, model=model, 
                                      criterion=criterion, optimizer=optimizer, 
                                      scheduler=lr_scheduler, device=device)

        valid_loss, valid_acc = evaluate(dataloader=valid_dataloader, model=model, 
                                         criterion=criterion, device=device)
        
        epoch_train_loss = np.mean(train_loss)
        epoch_train_acc = np.mean(train_acc)
        epoch_valid_loss = np.mean(valid_loss)
        epoch_valid_acc = np.mean(valid_acc)
        
        # my addition
        train_losses.append(epoch_train_loss)
        train_accs.append(epoch_train_acc)
        valid_losses.append(epoch_valid_loss)
        valid_accs.append(epoch_valid_acc)

        # Save the model that achieves the smallest validation loss.
        if epoch_valid_loss < best_valid_loss:
            best_val_acc = epoch_valid_acc
            if not os.path.exists(CHECKPOINT_FOLDER):
               os.makedirs(CHECKPOINT_FOLDER)
            print("Saving ...")
            state = {'state_dict': model.state_dict(),
                    'epoch': epoch}
            torch.save(state, os.path.join(CHECKPOINT_FOLDER, 'hw3_LSTM.pth'))
            # Your code: save the best model somewhere (no need to submit it to Sakai)

        print(f'epoch: {epoch+1}')
        print(f'train_loss: {epoch_train_loss:.3f}, train_acc: {epoch_train_acc:.3f}')
        print(f'valid_loss: {epoch_valid_loss:.3f}, valid_acc: {epoch_valid_acc:.3f}')


    # Your Code: Load the best model's weights.
    
    state_dict = torch.load(os.path.join(CHECKPOINT_FOLDER + 'hw3_LSTM.pth'))['state_dict']
    model.load_state_dict(state_dict)
    model = model.cuda()
    
    # Your Code: evaluate test loss on testing dataset (NOT Validation)
    test_loss, test_acc = evaluate(dataloader=test_dataloader, model=model, 
                                   criterion=criterion, device=device)

    epoch_test_loss = np.mean(test_loss)
    epoch_test_acc = np.mean(test_acc)
    print(f'test_loss: {epoch_test_loss:.3f}, test_acc: {epoch_test_acc:.3f}')
    
    # Free memory for later usage.
    del model
    torch.cuda.empty_cache()
    return {
        'num_params': num_params,
        "test_loss": epoch_test_loss,
        "test_acc": epoch_test_acc,
        "train_loss": train_losses, 
        "train_accs": train_accs, 
        "valid_loss": valid_losses, 
        "valid_accs": valid_accs
    }


def plot_train_val_acc(save_name, ret_dict, orig_hp, new_hp, save, gru=False):

    fig, ax = plt.subplots(1, 1)
    xx = np.linspace(1, orig_hp.N_EPOCHS, orig_hp.N_EPOCHS)

    ax.plot(xx, ret_dict['train_accs'], label='Train')
    ax.plot(xx, ret_dict['valid_accs'], 
            label='Validation; max accuracy=%.4f' % (np.max(ret_dict['valid_accs'])))
    ax.set_ylim([0.45, 1.05])

    diff_param = diff_hparams(orig_hp=orig_hp, new_hp=new_hp)
    
    lstm_or_gru = {False: 'LSTM', True: 'GRU'}
    
    if len(diff_param) != 0:
        title_ = lstm_or_gru[gru] + ';' + diff_param
    else:
        title_ = lstm_or_gru[gru]
    title_ += ';test_acc:%.4f' % ret_dict['test_acc']
    
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.set_title(title_)
    fig.tight_layout()
    if save == 'y':
        if len(diff_param) != 0:
            plt.savefig('%s_%s.pdf' % (save_name, re.sub(r'[^a-zA-Z0-9,=]', '', diff_param)), dpi=500, bbox_inches='tight')
        else:
            plt.savefig('%s_default_hp.pdf' % (save_name), dpi=500, bbox_inches='tight')
    return 


