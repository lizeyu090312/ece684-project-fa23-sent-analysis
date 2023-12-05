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

from models import *

from typing import Tuple, Dict, List


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


def build_vocab(x_train:list, min_freq: int=5, hparams: HyperParams =None) -> dict:
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
    # corpus_ = sorted(corpus, key=corpus.get, reverse=True)[:1000]
    corpus_ = [word for word, freq in corpus.items() if freq >= min_freq]
    # creating a dict
    vocab = {word : idx + 2 for idx, word in enumerate(corpus_)}
    vocab[hparams.PAD_TOKEN] = hparams.PAD_INDEX
    vocab[hparams.UNK_TOKEN] = hparams.UNK_INDEX
    return vocab


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

    
def collate(batch, pad_index):
    batch_ids = [torch.LongTensor(i['ids']) for i in batch]
    batch_ids = nn.utils.rnn.pad_sequence(batch_ids, padding_value=pad_index, batch_first=True)
    batch_length = torch.Tensor([i['length'] for i in batch])
    batch_label = torch.LongTensor([i['label'] for i in batch])
    batch = {'ids': batch_ids, 'length': batch_length, 'label': batch_label}
    return batch


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


# def predict_sentiment(text, model, vocab, device):
#     tokens = tokenize(vocab, text)
#     ids = [vocab[t] if t in vocab else UNK_INDEX for t in tokens]
#     length = torch.LongTensor([len(ids)])
#     tensor = torch.LongTensor(ids).unsqueeze(dim=0).to(device)
#     prediction = model(tensor, length).squeeze(dim=0)
#     probability = torch.softmax(prediction, dim=-1)
#     predicted_class = prediction.argmax(dim=-1).item()
#     predicted_probability = probability[predicted_class].item()
#     return predicted_class, predicted_probability

collate_fn = collate


def train_and_test_model_with_hparams(hparams, model_type="lstm", **kwargs):
    CHECKPOINT_FOLDER = "./saved_model/"
    # computes x_train, x_valid... etc once only for all models
    x_train, x_valid, x_test, y_train, y_valid, y_test = load_imdb()
    # Seeding. DO NOT TOUCH! DO NOT TOUCH hparams.SEED!
    # Set the random seeds.
    torch.manual_seed(hparams.SEED)
    random.seed(hparams.SEED)
    np.random.seed(hparams.SEED)

    vocab = build_vocab(x_train, hparams=hparams)

    vocab_size = len(vocab)
    print(f'Length of vocabulary is {vocab_size}')
    # use x_train, x_valid, x_test, y_train, y_valid, y_test as global
    # better for comparing betw models.
    train_data = IMDB(x_train, y_train, vocab, hparams.MAX_LENGTH)
    valid_data = IMDB(x_valid, y_valid, vocab, hparams.MAX_LENGTH)
    test_data = IMDB(x_test, y_test, vocab, hparams.MAX_LENGTH)

    collate = functools.partial(collate_fn, pad_index=hparams.PAD_INDEX)

    train_dataloader = torch.utils.data.DataLoader(
        train_data, batch_size=hparams.BATCH_SIZE, collate_fn=collate, shuffle=True)
    valid_dataloader = torch.utils.data.DataLoader(
        valid_data, batch_size=hparams.BATCH_SIZE, collate_fn=collate)
    test_dataloader = torch.utils.data.DataLoader(
        test_data, batch_size=hparams.BATCH_SIZE, collate_fn=collate)
    
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


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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