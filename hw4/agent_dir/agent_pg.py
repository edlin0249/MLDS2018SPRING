import math
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import scipy.misc
import numpy as np


def save_checkpoint(items, names, filename):
    state = {}
    for item, name in zip(items, names):
        state[name] = item.state_dict()
    
    torch.save(state, filename)


class Model(torch.nn.Module):
    def __init__(self, observ_dim, action_num):
        super(Model, self).__init__()
        self.cnn1 = nn.Conv2d(1, 16, 8, stride=4)
        self.cnn2 = nn.Conv2d(16, 32, 4, stride=2)
        self.linear1 = nn.Linear(2048, 128)
        self.linear2 = nn.Linear(128, action_num)
        
    def forward(self, frames):
        frames = frames.unsqueeze(-3)
        x = F.relu(self.cnn1(frames))
        x = F.relu(self.cnn2(x))
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.linear1(x))
        x = F.softmax(self.linear2(x), dim=1)
        
        return x


class Agent_PG():
    def __init__(self, env, args):
        self.env = env
        self.model = Model(self.env.observation_space.shape, self.env.action_space.n).cuda()
        self.opt = torch.optim.RMSprop(self.model.parameters(), lr=1e-4)
        torch.cuda.manual_seed_all(0)
        self.prev = 0
        
        checkpoint = torch.load(args.model_pg)
        self.model.load_state_dict(checkpoint['model'])

    def init_game_setting(self):
        pass

    def preprocessing(self, observation):
        observation = observation[34:194]
        observation[observation[:, :, 0] == 144] = 0
        observation[observation[:, :, 0] == 109] = 0
        observation = 0.2126 * observation[:, :, 0] + 0.7152 * observation[:, :, 1] + 0.0722 * observation[:, :, 2]
        observation = observation.astype(np.uint8)
        observation = scipy.misc.imresize(observation, (80, 80)).astype(float)
        processed = observation - self.prev
        self.prev = observation
        return processed

    def train(self):
        iteration = 1
        total_scores = [-1.0]
        while iteration < 2e7:
            mean = sum(total_scores[-2000:]) / len(total_scores[-2000:])
            
            observation = self.env.reset()
            self.opt.zero_grad()

            scores = [0]
            done = False
            while not done:
                log_probs = 0
                while scores[-1] == 0:
                    observation_torch = Variable(torch.from_numpy(self.preprocessing(observation)).float().unsqueeze(0)).cuda()
                    probabilities = self.model(observation_torch)
                    sampled = torch.multinomial(probabilities, 1).data[0, 0]

                    observation, score, done, _ = self.env.step(sampled)

                    scores[-1] += score
                    log_probs += probabilities[:, sampled].log()

                loss = -(score - mean) * log_probs
                loss.backward()
                scores.append(0)

            torch.nn.utils.clip_grad_norm(self.model.parameters(), 5, 'inf')
            self.opt.step()
            
            print('score: ' + str(sum(scores)))
            total_scores += scores
            
            if iteration % 10 == 9:
                save_checkpoint([self.model], ['model'], 'model_pg')

            iteration += 1
            
    def make_action(self, observation, test=True):
        observation_torch = Variable(torch.from_numpy(self.preprocessing(observation)).float().unsqueeze(0)).cuda()
        probabilities = self.model(observation_torch)
        action = torch.multinomial(probabilities, 1).data[0, 0]
        return action
