#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 16:08:21 2018

@author: afvk
"""

# -*- coding: utf-8 -*-
import numpy as np
import random
from collections import deque
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected


class HPS:
    
    def __init__(self, gamma, epsilon, epsilon_min, epsilon_decay, lr,
                 memory_size):
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.lr = lr
        self.memory_size = memory_size
        

class Agent:
    
    def __init__(self, state_size, action_size, hps):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=hps.memory_size)
        self.hps = hps
        self.model = Model(state_size, action_size, hps.lr)
    
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))


    def act(self, state):
        if np.random.rand() <= self.hps.epsilon:
            return np.random.randint(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])


    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.hps.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
            
        if self.hps.epsilon > self.hps.epsilon_min:
            self.hps.epsilon *= self.hps.epsilon_decay


    def save(self, name):
        self.model.save(name)


    def load(self, name):
        self.model.load(name)


 
class Model:
    
    def __init__(self, state_size, action_size, lr):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr
        
        self.build()
        
    
    def build(self):
        self.inputs = tf.placeholder(tf.float32, shape=(None,self.state_size))
        
        x = fully_connected(self.inputs, 24, activation_fn=tf.nn.relu)
        x = fully_connected(x, 24, activation_fn=tf.nn.relu)
        self.predictions = fully_connected(x, self.action_size, 
                                           activation_fn=None)
        
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        
        self.labels = tf.placeholder(tf.float32, shape=(None,self.action_size))
        loss = tf.losses.mean_squared_error(labels=self.labels, 
                                            predictions=self.predictions)
        
        self.update = optimizer.minimize(loss)


    def predict(self, state):
        Q = self.sess.run(self.predictions, feed_dict={self.inputs:state})
        return Q


    def fit(self, state, labels, epochs, verbose):
        self.sess.run(self.update, feed_dict={self.inputs:state,
                                                  self.labels:labels})
    
    
    def save(self, fn):
        print('Cannot save %s, not implemented yet'%fn)
    
    
    def load(self, fn):
        print('Cannot load %s, not implemented yet'%fn)
        
        









