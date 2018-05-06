import os
import gc
import sys
import json
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

test_path = sys.argv[1]
output_path = sys.argv[2]
dictionary_path = sys.argv[3]
model_path = sys.argv[4]


def clear_memory():
    gc.collect()
    torch.cuda.empty_cache()


# DEFINE DICTIONARY
SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"SOS":0, "EOS":1, "UNK":2, "PAD":3}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNK", 3:"PAD"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        sentence = sentence.replace('.', '')
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def filterPairs(pairs):
    return [pair for pair in pairs]

def indexesFromSentence(lang, sentence):
    sentence = sentence.replace('.', '')
    index = []
    for word in sentence.split(' '):
        if word in lang.index2word.values():
            index.append(lang.word2index[word])
        else:
            index.append(lang.word2index['UNK'])
    return index

def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    result = torch.LongTensor(indexes).view(-1, 1)
    return result


# LOAD DICTIONARY
with open(dictionary_path, 'rb') as handle:
    lang_train = pickle.load(handle)


# LOAD TESTING ID
id_path = os.path.join(test_path, 'id.txt')
with open(id_path, 'r') as handle:
    keys = handle.read().split('\n')[:-1]


# LOAD TESTING DATA
testing_data_dict = {}
for key in keys:
    filename = os.path.join(test_path, 'feat', key + '.npy')
    test_data_temp = np.load(filename)
    testing_data = torch.FloatTensor(test_data_temp)
    testing_data_dict[key] = testing_data


# DEFINE MODELS: ENCODER
class EncoderRNN(nn.Module):
    def __init__(self, num_layers=1, hidden_dim=512, feature_dim=4096, input_dim=1024):
        super(EncoderRNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.feature_dim = feature_dim
        self.reduction = nn.Linear(feature_dim, input_dim)
        self.gru = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, features, hiddens):
        # features: (batch, seq_len, feature_dim)
        # hiddens: (num_layers * num_directions, batch, hidden_dim)
        # output: (batch, seq_len, hidden_size * num_directions)
        
        reducted = self.reduction(features)
        output, hiddens = self.gru(reducted, hiddens)
        return output, hiddens

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


# DEFINE MODELS: DECODER
class AttnDecoderRNN(nn.Module):
    def __init__(self, token_num, embedding_dim, decoder_dim, encoder_dim, input_dim, 
                 in_seqlen, num_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        
        # token_num: total num of words, used for word embedding
        # embedding_dim: dim of word-embedding result
        # decoder_dim: dim of hidden layer of decoder
        # encoder_dim: dim of previous encoder's hidden layers
        # input_dim: dim of GRU's input
        # num_layers: how deep the GRU is
        # in_seqlen: input sequence length, used for attention
        
        self.num_layers = num_layers
        self.decoder_dim = decoder_dim
        
        self.embedding = nn.Embedding(token_num, embedding_dim, padding_idx=lang_train.word2index['PAD'])
        self.dropout = nn.Dropout(dropout_p)
        
        self.attn = nn.Linear(embedding_dim + decoder_dim, in_seqlen)
        self.attn_combine = nn.Linear(embedding_dim + encoder_dim, input_dim)
        
        self.gru = nn.GRU(input_size=input_dim, hidden_size=decoder_dim, num_layers=self.num_layers)
        self.out = nn.Linear(decoder_dim, token_num)

    def forward(self, batch_tokens, encoder_outputs, hiddens):
        # batch tokens: (batch,)
        # encoder_outputs: (batch, seq_len, hidden_size)
        # hiddens: (num_layers * num_directions, batch, hidden_dim)

        embedded_input = self.embedding(batch_tokens)
        embedded_input = self.dropout(embedded_input)
        # embedded_input: (batch, embedding_dim)
            
        attn_depend = torch.cat((embedded_input, hiddens[0]), dim=1)
        attn_weights = F.softmax(self.attn(attn_depend), dim=1)
        # attn_weights: (batch, in_seqlen)
        
        # (B, 1, SEQLEN) * (B, SEQLEN, HIDDEN_DIM) = (B, 1 HIDDEN_DIM) => (B, HIDDEN_DIM)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        # attn_applied: (batch, encoder_dim)
        
        gru_input = torch.cat((embedded_input, attn_applied), dim=1)
        gru_input = self.attn_combine(gru_input)
        gru_input = F.relu(gru_input)
        gru_input = gru_input.unsqueeze(0)
        # gru_input: (1, batch, input_dim)

        gru_output, hiddens = self.gru(gru_input, hiddens)
        # gru_output: (1, batch, hidden_size)
        # hiddens: (num_layers * num_directions, batch, hidden_dim)
        
        output = self.out(gru_output[0])
        output = F.log_softmax(output, dim=1)
        # output: (batch, token_num), as distribution
        
        return output, hiddens, attn_weights

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim)


# CREATE MODEL
encoder = EncoderRNN(feature_dim=4096, input_dim=512, hidden_dim=256).cuda()
decoder = AttnDecoderRNN(token_num=lang_train.n_words, embedding_dim=256, decoder_dim=256, 
                        encoder_dim=256, input_dim=512, in_seqlen=80).cuda()


# LOAD MODEL
checkpoint = torch.load(model_path)
encoder.load_state_dict(checkpoint['encoder_state'])
decoder.load_state_dict(checkpoint['decoder_state'])


# CLEAR MEMORY
del checkpoint
clear_memory()


# DEFINE NODE FOR BEAM SEARCH
class Node:
    def __init__(self, token, hidden, loss, prev):
        self.token = token
        self.hidden = hidden
        self.loss = loss
        self.prev = prev
    
    def getchar(self):
        return lang_train.index2word[self.token]
    
    def getpath(self):
        now = self
        path = []
        while now.token != lang_train.word2index['SOS']:
            if now.token == lang_train.word2index['EOS'] or now.token == lang_train.word2index['PAD']:
                now = now.prev
                continue
            path.append(str(lang_train.index2word[now.token]))
            now = now.prev
        
        return ' '.join(list(reversed(path)))


# DEFINE BEAM SEARCH PREDICTION
def infer_beam(data, encoder, decoder, outlen=15, beam=3):
    encoder.eval()
    decoder.eval()
    
    answers = []
    for index, key in enumerate(data.keys()):
        X = [ data[key] ]
        X = torch.stack(X)
        X = Variable(X.cuda(), volatile=True)

        # ENCODE STAGE    
        encoder_hiddens = Variable(encoder.initHidden(1).cuda(), volatile=True)
        encoder_outputs, encoder_hiddens = encoder.forward(X, encoder_hiddens)
        # encoder_outputs: (batch, seq_len, hidden_size)

        # DECODE STAGE
        queue = [ Node(SOS_token, encoder_hiddens.cpu(), 0, None) ]
        timestamp = 1
        while len(queue) != 0:
            nowleaf = queue.pop(0)
            token = Variable(torch.LongTensor([nowleaf.token]).cuda(), volatile=True)
            output, new_hiddens, attn_weights = decoder.forward(token, encoder_outputs, nowleaf.hidden.cuda())
            max_value, max_index = output[0].topk(beam)
            for value, index in zip(max_value.data.cpu(), max_index.data.cpu()):
                queue.append( Node(int(index), new_hiddens.cpu(), nowleaf.loss + value, nowleaf) )

            if len(queue) == beam * beam:
                queue = sorted(queue, key=lambda node: node.loss, reverse=True)
                queue = queue[:beam]
                timestamp += 1
                if timestamp == outlen:
                    break
        
        answers.append(queue[0].getpath())
        
    return answers


# PREDICT
answers = infer_beam(testing_data_dict, encoder, decoder)
clear_memory()


with open(output_path, 'w') as handle:
    for key, answer in zip(keys, answers):
        print(key + ',' + answer, file=handle)
