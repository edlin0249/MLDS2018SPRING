import gc
import os
import sys
import json
import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


train_data_path = sys.argv[1]
train_label_path = sys.argv[2]
train_epoch = int(sys.argv[3])
model_save_path = sys.argv[4]


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since):
    now = time.time()
    s = now - since
    return '%s' % (asMinutes(s))


def save_checkpoint(encoder, decoder, encoder_opt, decoder_opt, filename):
    state = {
        'encoder_state': encoder.state_dict(),
        'decoder_state': decoder.state_dict(),
        'encoder_opt': encoder_opt.state_dict(),
        'decoder_opt': decoder_opt.state_dict()
    }
    torch.save(state, filename)


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


print('Loading training labels...')
with open(train_label_path) as handle:
    training_label = json.load(handle)


print('Creating dictionary...')
lang_train = Lang('train')
for label in training_label:
    for sentence in label['caption']:
        lang_train.addSentence(sentence)

training_label_dict = {}
training_label_sentence_dict = {}
for label in training_label:
    sens = []
    sentenses = []
    for sentence in label['caption']:
        sens.append(variableFromSentence(lang_train, sentence))
        sentenses.append(sentence)
    training_label_dict[label['id']] = sens
    training_label_sentence_dict[label['id']] = sentenses


print('Loading training data')
training_data_dict = {}
for key in training_label_dict.keys():
    filename = os.path.join(train_data_path, 'feat', key + '.npy')
    train_data_temp = np.load(filename)
    training_data = torch.FloatTensor(train_data_temp)
    training_data_dict[key] = training_data


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




def train(input_seq, target_seq, encoder, decoder, encoder_opt,
            decoder_opt, criterion, outlen=10, teacher_forcing_ratio=1):
    # input_seq: (batch, seq_len, feature_dim)
    # target_seq: (batch, outlen)
    batch_size = input_seq.size()[0]
    teacher_forcing = np.random.random() < teacher_forcing_ratio
    
    # ENCODE STAGE    
    encoder_opt.zero_grad()
    encoder_hiddens = Variable(encoder.initHidden(batch_size).cuda())
    encoder_outputs, encoder_hiddens = encoder.forward(input_seq, encoder_hiddens)
    # encoder_outputs: (batch, seq_len, hidden_size)

    # DECODE STAGE
    loss = 0
    decoder_opt.zero_grad()
    batch_tokens = Variable(torch.LongTensor([SOS_token] * batch_size).cuda())
    decoder_hiddens = encoder_hiddens
    for t in range(outlen):
        output, decoder_hiddens, attn_weights = decoder.forward(batch_tokens, encoder_outputs, decoder_hiddens)
        slot = criterion(output, target_seq[:, t])
        loss += slot
        max_value, max_index = output.topk(1, dim=1)
        batch_tokens = max_index.view(-1)
        
        if teacher_forcing:
            batch_tokens = target_seq[:, t]
    
    loss.backward()
    encoder_opt.step()
    decoder_opt.step()
    
    return loss.data.cpu()[0]


def padding(data, length):
    if len(data) < length:
        pad = torch.LongTensor([lang_train.word2index['PAD']] * (length - len(data)))
        return torch.cat((data, pad))
    else:
        return data[:length]


def trainIters(encoder, decoder, encoder_opt, decoder_opt, batch=32, epoch=10, print_every=1, history=[]):
    start = time.time()
    loss_total = 0
    training_count = 0
    outlen = 10
    sample_num = 0
    
    encoder.train()
    decoder.train()
    
    keys = list(training_label_dict.keys())
    criterion = nn.NLLLoss()
    for iteration in range(epoch):
        print('EPOCH ' + str(iteration))
        np.random.shuffle(keys)
        for batch_num, index in enumerate(range(0, len(keys), batch)):
            X = []
            Y = []
            for i, key in enumerate(keys[index : min(index + batch, len(keys))]):
                targets = training_label_dict[key]
                # Y.append( padding(targets[np.random.randint(len(targets))].view(-1), outlen) )
                Y.append( padding(targets[0].view(-1), outlen) )
                X.append( training_data_dict[key] )
            X = torch.stack(X)
            Y = torch.stack(Y)
            X = Variable(X.cuda())
            Y = Variable(Y.cuda())
            
            loss = train(X, Y, encoder, decoder, encoder_opt, decoder_opt, criterion)
            loss_total += loss * len(Y)
            sample_num += len(Y)
            
            if batch_num % print_every == print_every - 1:
                loss_avg = loss_total / sample_num
                sample_num = 0
                loss_total = 0
                history.append(loss_avg)            
                print('%s (%d %d%%) | LOSS: %.4f' % (timeSince(start), index, index / 1450 * 100, loss_avg))
    
    return history


print('Creating model...')
encoder = EncoderRNN(feature_dim=4096, input_dim=512, hidden_dim=256).cuda()
decoder = AttnDecoderRNN(token_num=lang_train.n_words, embedding_dim=256, decoder_dim=256, 
                        encoder_dim=256, input_dim=512, in_seqlen=80).cuda()

encoder_opt = optim.Adam(encoder.parameters())
decoder_opt = optim.Adam(decoder.parameters())

print('Start training...')
history = trainIters(encoder, decoder, encoder_opt, decoder_opt, 
                        batch=100, epoch=train_epoch, print_every=2)

print('Saving model...')
save_checkpoint(encoder, decoder, encoder_opt, decoder_opt, model_save_path)

print('Training finished')
