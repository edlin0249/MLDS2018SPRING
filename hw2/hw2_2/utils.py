from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np
import pickle

use_cuda = torch.cuda.is_available()
print(use_cuda)

def words2sentence(words):
    output = ''
    for word in words:
        output += ' '
        output += word
    return output

import numpy as np
import sys
import codecs
import os
import math
import operator
import json
from functools import reduce
import time


def fetch_data(cand, ref):
    """ Store each reference and candidate sentences as a list """
    references = []
    if '.txt' in ref:
        reference_file = codecs.open(ref, 'r', 'utf-8')
        references.append(reference_file.readlines())
    else:
        for root, dirs, files in os.walk(ref):
            for f in files:
                reference_file = codecs.open(os.path.join(root, f), 'r', 'utf-8')
                references.append(reference_file.readlines())
    candidate_file = codecs.open(cand, 'r', 'utf-8')
    candidate = candidate_file.readlines()
    return candidate, references


def count_ngram(candidate, references, n):
    clipped_count = 0
    count = 0
    r = 0
    c = 0
    for si in range(len(candidate)):
        # Calculate precision for each sentence
        ref_counts = []
        ref_lengths = []
        # Build dictionary of ngram counts
        for reference in references:
            ref_sentence = reference[si]
            ngram_d = {}
            words = ref_sentence.strip().split()
            ref_lengths.append(len(words))
            limits = len(words) - n + 1
            # loop through the sentance consider the ngram length
            for i in range(limits):
                ngram = ' '.join(words[i:i+n]).lower()
                if ngram in ngram_d.keys():
                    ngram_d[ngram] += 1
                else:
                    ngram_d[ngram] = 1
            ref_counts.append(ngram_d)
        # candidate
        cand_sentence = candidate[si]
        cand_dict = {}
        words = cand_sentence.strip().split()
        limits = len(words) - n + 1
        for i in range(0, limits):
            ngram = ' '.join(words[i:i + n]).lower()
            if ngram in cand_dict:
                cand_dict[ngram] += 1
            else:
                cand_dict[ngram] = 1
        clipped_count += clip_count(cand_dict, ref_counts)
        count += limits
        r += best_length_match(ref_lengths, len(words))
        c += len(words)
    if clipped_count == 0:
        pr = 0
    else:
        pr = float(clipped_count) / count
    bp = brevity_penalty(c, r)
    return pr, bp

def clip_count(cand_d, ref_ds):
    """Count the clip count for each ngram considering all references"""
    count = 0
    for m in cand_d.keys():
        m_w = cand_d[m]
        m_max = 0
        for ref in ref_ds:
            if m in ref:
                m_max = max(m_max, ref[m])
        m_w = min(m_w, m_max)
        count += m_w
    return count

def best_length_match(ref_l, cand_l):
    """Find the closest length of reference to that of candidate"""
    least_diff = abs(cand_l-ref_l[0])
    best = ref_l[0]
    for ref in ref_l:
        if abs(cand_l-ref) < least_diff:
            least_diff = abs(cand_l-ref)
            best = ref
    return best

def brevity_penalty(c, r):
    if c > r:
        bp = 1
    else:
        bp = math.exp(1-(float(r)/c))
    return bp

def geometric_mean(precisions):
    return (reduce(operator.mul, precisions)) ** (1.0 / len(precisions))

def BLEU(candidate, references):
    precisions = []
    for i in range(4):
        pr, bp = count_ngram(candidate, references, i+1)
        precisions.append(pr)
    bleu = geometric_mean(precisions) * bp
    return bleu

def v2c(x):
    '''
    if use gpu then return x.cuda
    '''
    if use_cuda:
        return x.cuda()
    else:
        return x


SOS_token = 0
EOS_token = 1
PAD_token = 3
UNK_token = 2
class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"<SOS>":0, "<EOS>":1, "<UNK>":2, "<PAD>":3}
        self.word2count = {}
        self.index2word = {0: "<SOS>", 1: "<EOS>", 2:"<UNK>", 3:"<PAD>"}
        self.n_words = 4  # Count SOS and EOS

    def addSentence(self, sentence):
        #sentence = sentence.replace('.', '')
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

def indexesFromSentence(lang, sentence, max_length):
    index = []
    for word in sentence.split(' '):
        if word in lang.index2word.values():
            index.append(lang.word2index[word])
        else:
            index.append(UNK_token)
      
    index.append(EOS_token)
     
    if len(index) < max_length:
        index += [PAD_token] * (max_length - len(index))
    else:
        index = index[:max_length]
      
    return torch.LongTensor(index)

def variableFromSentence(lang, sentence, max_length):
    indexes = indexesFromSentence(lang, sentence, max_length)
    
    result = Variable(torch.LongTensor(indexes).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def sens2tensor(sens, lang):
    max_len = 0
    for sen in sens:
        max_len = max(max_len, len(sen.split(' ')))
        
    sen_indexes = []
    for sen in sens:
        sen_indexes.append(indexesFromSentence(lang, sen, max_len))
        
    sen_indexes = np.array(sen_indexes)
    tensor = []
    for n in range(max_len + 1):
        sen = Variable(torch.LongTensor(sen_indexes[:, n].reshape((1, -1))))
        sen = v2c(sen)
        tensor.append(sen)
    
    return tensor

def get_dataset(file_dict, batch_size = 8):
    lang_txt = Lang('train_txt')
    lines = open(file_dict).read().strip().split('\n')
    for sen in lines:
        lang_txt.addSentence(sen)
    
    len_lines = len(lines)
    
    dataset = []
    
    for index, sen in enumerate(lines):
        if(index != (len_lines - 1)):
            next_sen = lines[index + 1]
            pair = (sen, next_sen)
            max_len = max(len(sen.split(' ')), len((next_sen.split(' '))))
            dataset.append((max_len, pair))
    dataset = sorted(dataset)
    
    print('dataset prepared')
    
    N = len(dataset)
    batch_num = int(np.ceil(N/batch_size))
    
    batch_data = [None] * batch_num 
    for n in range(batch_num):
        batch_data[n] = ([],[])
        
    for index, ele in enumerate(dataset):
        
        if index % 20000 == 0:
            print('completed', index/(N+ 0.0))
        
        _, pair = ele
        train_sen , label_sen = pair
        batch_index = int(index/batch_size)
        sens, labels = batch_data[batch_index]
        sens.append(train_sen)
        labels.append(label_sen)
    
    print('split prepared')
    
    output_data = []
    for index, ele in enumerate(batch_data):
        if index % 200 == 0:
            print('finished', index/(batch_num + 0.0))
        train_sens, label_sens = ele
        train_sens_tensor = sens2tensor(train_sens, lang_txt)
        label_sens_tensor = sens2tensor(label_sens, lang_txt)
        output_data.append((train_sens_tensor, label_sens_tensor, label_sens))
     
    return output_data, lang_txt

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

"""
file_dict = 'clr_conversation.txt' ###########the file name##############
batch_data,lang_txt = get_dataset(file_dict, batch_size = 64)
print(lang_txt.n_words)
voc_file = open('voc_hw2_2','wb') 
pickle.dump(lang_txt, voc_file, 0)
voc_file.close()
"""
class EncoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, output_length):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_length, feature_size)
        self.gru = nn.GRU(feature_size, hidden_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = torch.zeros(1, batch_size, self.hidden_size)
        return result

MAX_LENGTH = 30
class DecoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, output_length):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_length = output_length
        self.feature_size = feature_size
        
        self.embedding = nn.Embedding(output_length, feature_size)
        self.gru = nn.GRU(feature_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_length)
        self.softmax = nn.LogSoftmax(dim=2)
        
    def forward(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output))
        return output, hidden
      
    def initHidden(self, batch_size):
        result = torch.zeros(1, batch_size, self.hidden_size)
        return result  

feature = 128
#encoder = EncoderRNN(feature, feature, lang_txt.n_words)                     
#decoder = DecoderRNN(feature, feature,  lang_txt.n_words)
"""
if use_cuda:
    encoder = encoder.cuda()
    
    decoder = decoder.cuda()
"""
