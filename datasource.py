import numpy as np
import torch

def get_data_random(timesteps=1):
    d = np.random.randint(low=0, high=100, size=(timesteps, 1))
    return d.copy()

def get_data_random_demand():
    pass

def get_data_sine_wave(timesteps=1):
    d = np.sin(np.linspace(0, 10*np.pi, timesteps))    
    return d
