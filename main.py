#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:22:25 2018

@author: arent
"""

import gym
from dqn import Agent, HPS

import pickle
import numpy as np

import tensorflow as tf

N_eps = 1000

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995
lr = 0.001
memory_size = 2000

eps_watch = [5, 10, 15, 20]
save_freq = 10

hps = HPS(gamma, epsilon, epsilon_min, epsilon_decay, lr, memory_size)

env = gym.make('CartPole-v1')
state_size = env.observation_space.shape[0]
action_size = env.action_space.n
agent = Agent(state_size, action_size, hps)

done = False
batch_size = 32

results = dict()
results['times'] = []
results['episodes'] = []

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    agent.model.sess = sess
    
    for eps in range(N_eps):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        
        for time in range(500):
            if eps in eps_watch:
                env.render()
                
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            
            if done:
                results['times'].append(time)
                results['episodes'].append(eps)
                
                msg = "episode: %i/%i, \tscore: %i, \teps: %.3f"
                print(msg%(eps, N_eps, time, agent.hps.epsilon))
                break
            
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
#        if eps in eps_watch:
#            env.close()
            
        if eps%save_freq == 0:
            pickle.dump(results, open('./save/results.p', 'wb'))












