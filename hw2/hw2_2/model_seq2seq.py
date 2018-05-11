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


import numpy as np
import sys
import codecs
import os
import math
import operator
import json
from functools import reduce
import time

import matplotlib.pyplot as plt
import sys

#from utils import *

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
    """
    if use gpu then return x.cuda
    """
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
    #sentence = sentence.replace('.', '')
    index = []
    for word in sentence.split(' '):
        if word in lang.index2word.values():
            index.append(lang.word2index[word])
        else:
            index.append(UNK_token)
      
    len_indexes = len(index)
  
    index.append(EOS_token)
  
    for _ in range(max_length - len_indexes):
        index.append(PAD_token)  
      
    return index

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


file_dict = sys.argv[1] ###########the file name##############
batch_data,lang_txt = get_dataset(file_dict, batch_size = 64)
print(lang_txt.n_words)
voc_file = open('voc_hw2_2','wb') 
pickle.dump(lang_txt, voc_file, 0)
voc_file.close()

class EncoderRNN(nn.Module):
    def __init__(self, feature_size, hidden_size, output_length):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_length, feature_size)
        self.gru = nn.GRU(feature_size, hidden_size)
        
    def forward(self, input, hidden):
        embedded = self.embedding(input.view(1, -1))
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

MAX_LENGTH = 100
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
        
    def forward(self, input, hidden, rubbish):
        output = self.embedding(input.view(1, -1))
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output))
        return output, hidden, None
      
    def initHidden(self, batch_size):
        result = Variable(torch.zeros(1, batch_size, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result  

feature = 128
encoder = EncoderRNN(feature, feature, lang_txt.n_words)                     
decoder = DecoderRNN(feature, feature,  lang_txt.n_words)

if use_cuda:
    encoder = encoder.cuda()
    
    decoder = decoder.cuda()


class seq2seq(nn.Module):
  
    def __init__(self,
               encoder,
               decoder,
              
               max_length = MAX_LENGTH,
               
               decay_rate = 1.0
    ):
        super(seq2seq,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
   
        self.learning_rate = 0.001
    
        self.hidden_size = encoder.hidden_size
    
        self.criterion = nn.NLLLoss()
        self.evaluater_criterion = nn.MSELoss()
    
        self.encoder_optimizer = optim.Adam(encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(decoder.parameters(), lr=self.learning_rate)
    
    
        self.max_sentence_length = 0
    
    def random_sample(self,tensor):
            
        tensor1 = (torch.exp(tensor[0])).data.cpu().numpy()
        a,b = tensor1.shape
        samples = np.zeros((1, a))
        probs = np.zeros((1, a))
        for i in range(a):
            max_index = tensor1[i].argmax()
            tensor1[i, max_index] += (1 - tensor1[i].sum())
            
            index = np.random.choice(b, 1, p = tensor1[i])
            samples[:, i] = index
            probs[:, i] = tensor1[i,index]
        samples = v2c(Variable(torch.LongTensor(samples)))
        probs = v2c(Variable(torch.FloatTensor(probs)))
        return samples, probs

    def save_checkpoint(self, filename):
        state = {
            'encoder_state': self.encoder.state_dict(),
            'decoder_state': self.decoder.state_dict(),
            'encoder_opt': self.encoder_optimizer.state_dict(),
            'decoder_opt': self.decoder_optimizer.state_dict()
        }
        torch.save(state, filename)
  
    def trainIters(self, batch_data, XENT_iters, print_every=10, plot_every=2):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

      
        self.max_sentence_length = len(batch_data[0][0])
      
        training_count = 0
      
        n_iters = 0
        t_iters = XENT_iters 
        for iter in range(1, XENT_iters + 1):
            n_iters += 1
            for batch in batch_data:

                training_in, target, target_sentence = batch

                loss = self.train(
                    training_in,
                    target, 
                    target_sentence, 
                    train_method = 'XENT')

                print_loss_total += loss
                plot_loss_total += loss
                training_count += 1

                if training_count%print_every == 0:
                    print_loss_avg = print_loss_total / print_every
                    print_loss_total = 0
                    print('%s (%d %d%%) %.4f' % (timeSince(start, n_iters / t_iters),
                                             n_iters, n_iters / t_iters * 100, print_loss_avg))
                
                if training_count % plot_every == 0:
                    plot_loss_avg = plot_loss_total / plot_every
                    plot_losses.append(plot_loss_avg)
                    plot_loss_total = 0

        self.save_checkpoint('hw2_2.model')
                        
        return plot_losses
  
  
    def train(self,
              input_variable,
              target_variable,
              target_sentence, 
              max_length=MAX_LENGTH,
              train_method = 'XENT', 
              ):
    
        encoder = self.encoder
        decoder = self.decoder      
        encoder_optimizer = self.encoder_optimizer
        decoder_optimizer = self.decoder_optimizer      
        criterion = self.criterion
      
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

      
        batch_size = input_variable[0].size()[1]
      
        encoder_hidden = encoder.initHidden(batch_size)

      
        input_length = len(input_variable)
        target_length = len(target_variable)

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_variable[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0][0]
      
        start_token = np.ones((1, batch_size)) * SOS_token
        decoder_input = Variable(torch.LongTensor(start_token))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_hidden = encoder_hidden

        #use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

      
     
        if train_method == 'XENT':

            if True:
                # Teacher forcing: Feed the target as the next input
                for di in range(target_length):
                    decoder_output, decoder_hidden, decoder_attention = decoder(
                        decoder_input, decoder_hidden, encoder_outputs)
                    loss += criterion(decoder_output[0], target_variable[di][0])
                    decoder_input = target_variable[di]  # Teacher forcing
        
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        return loss.data[0] / target_length
 
test1 = seq2seq(encoder, decoder)
loss_actor = test1.trainIters(batch_data, 1) ########return the loss of every 50 batches
