import numpy as np
import torch
import random

# Dataset 1
def get_data_random(timesteps=1):
    d = np.random.randint(low=0, high=100, size=(timesteps, 1))
    return d.copy()

# Dataset 2
def get_data_random_poisson(lamda=5,timesteps=1):
    d = np.random.poisson(lam=lamda, size=timesteps)
    d = d.reshape(timesteps, 1)        
    return d.copy()

# Dataset 3
# Kaggle dataset
# Dataset 4
def get_data_random_demand(k=1, timesteps=1):
    d = []    
    for i in range(timesteps):
        dprime = np.array(random.choices([3, 4, 5, 6, 7, 8, 9], weights=(5, 5, 5, 10, 5, 20, 50), k=52))
        d.append(dprime)    
    d = np.array(d)
    d = d.reshape(k*timesteps, 1)
    return d


def get_data_sine_wave(timesteps=1):
    d = np.sin(np.linspace(0, 10*np.pi, timesteps))    
    return d

