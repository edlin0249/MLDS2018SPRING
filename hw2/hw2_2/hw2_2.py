import sys
import pickle

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import numpy as np

from utils import *

use_cuda = torch.cuda.is_available()

def evaluate(encoder, decoder, batch_sens, lang_txt, outlen=15):
    batch_size = len(batch_sens)
    max_in = len(batch_sens[-1][1].split())
    in_tensor = []
    for sens in batch_sens:
        in_tensor.append(indexesFromSentence(lang_txt, sens[1], max_in))
    in_tensor = torch.stack(in_tensor).t()  # (seqlen, batch_num)
    
    input_variable = Variable(in_tensor.cuda(), volatile=True) if use_cuda else Variable(in_tensor, volatile=True)
    encoder_hidden = Variable(encoder.initHidden(batch_size).cuda(), volatile=True)
    encoder_output, encoder_hidden = encoder(input_variable, encoder_hidden)
    
    decoder_hidden = encoder_hidden

    start_token = [SOS_token] * batch_size
    decoder_input = Variable(torch.LongTensor(start_token).view(1, -1), volatile=True)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    
    answers = np.empty((batch_size, outlen))
    for i in range(outlen):
        decoder_output, decoder_hidden = decoder(decoder_input, decoder_hidden)
        topv, topi = decoder_output[0].data.topk(1, dim=1)
        decoder_input[0, :] = topi[:, 0]
        answers[:, i] = topi[:, 0].cpu().numpy()
    
    answers = [(batch_sens[index][0], ''.join([lang_txt.index2word[char] for char in answer if char != EOS_token and char != PAD_token])) for index, answer in enumerate(answers)]
    return answers


voc_file = open('voc_hw2_2','rb')  
voc = pickle.load(voc_file)  
voc_file.close()

print('Loading model...')
checkpoint = torch.load('hw2_2.model')
encoder = EncoderRNN(feature, feature, voc.n_words)                     
decoder = DecoderRNN(feature, feature,  voc.n_words)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=0.001)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=0.001)

encoder.load_state_dict(checkpoint['encoder_state'])
decoder.load_state_dict(checkpoint['decoder_state'])
encoder_optimizer.load_state_dict(checkpoint['encoder_opt'])
decoder_optimizer.load_state_dict(checkpoint['decoder_opt'])

if use_cuda:
    encoder = encoder.cuda()
    decoder = decoder.cuda()
    
print('Reading test input...')
test_input = []
with open(sys.argv[1], 'r') as test_file:
    for line_id, sentence in enumerate(test_file):
        test_input.append((line_id, sentence[:-1]))
        
test_input_sorted = sorted(test_input, key=lambda x: len(x[1].split()))

print('Predicting...')
start_time = time.time()
with open(sys.argv[2], 'w') as result_output:
    batch_size = 100
    all_answers = []
    for index in range(0, len(test_input_sorted), batch_size):
        if (index / batch_size) % 10 == 0:
            print(str(index) + ' sentences predicted')
        
        batch_sens = test_input_sorted[index : min(index + batch_size, len(test_input_sorted))]
        answers = evaluate(encoder, decoder, batch_sens, voc)
        all_answers += answers
    all_answers = sorted(all_answers, key=lambda x: x[0])
    for index, answer in all_answers:
        result_output.write(answer + '\n')
