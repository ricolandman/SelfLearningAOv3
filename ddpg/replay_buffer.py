from collections import deque
import random
import matplotlib.pyplot as plt
import numpy as np

class ReplayBuffer(object):

    def __init__(self, buffer_size,episode_length, random_seed=123):
        self.buffer_size = buffer_size
        self.count = 0
        self.episode_length = episode_length
        self.buffer = deque()
        random.seed(random_seed)

    def add(self, h_s, h_a, h_r):
        experience = (h_s, h_a, h_r)
        if self.count < self.buffer_size:
            self.buffer.append(experience)
            self.count += 1
        else:
            self.buffer.popleft()
            self.buffer.append(experience)

    def size(self):
        return self.count

    def sample_batch(self, batch_size,N=1):
        batch = []

        episode_index = np.random.choice(np.arange(self.count),batch_size)
        start_index = np.random.choice(np.arange(N+1,self.episode_length-1),batch_size)

        indices = zip(episode_index,start_index)
        s_batch = [np.array([self.buffer[episode_i][0][t-N:t] for episode_i,t in \
                        zip(episode_index,start_index)]),
                   np.array([self.buffer[episode_i][1][t-N-1:t-1] for episode_i,t in \
                        zip(episode_index,start_index)])]

        a_batch = np.array([self.buffer[episode_i][1][t-N:t] for episode_i,t in \
                        zip(episode_index,start_index)])

        r_batch = np.array([self.buffer[episode_i][2][t-N:t] for episode_i,t in \
                        zip(episode_index,start_index)])

        s2_batch = [np.array([self.buffer[episode_i][0][t-N+1:t+1] for episode_i,t in \
                        zip(episode_index,start_index)]),
                   np.array([self.buffer[episode_i][1][t-N:t] for episode_i,t in \
                        zip(episode_index,start_index)])]

        return s_batch, a_batch, r_batch, s2_batch

    def clear(self):
        self.buffer.clear()
        self.count = 0
