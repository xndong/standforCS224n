# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:56:58 2019

@author: DongXiaoning
"""

import numpy as np

def softmax(x):       # vector
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def softmax1(x):
    e_x = np.exp(x)  
    return e_x / e_x.sum(axis=0)

def softmax2(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum(axis=0)
   
def sigmoid(x):       # scalar  
    return 1 / (1 + np.exp(-x))

def init_parameter(R_D,R_d,yt_row):
    np.random.seed(1)
    Whh = np.random.randn(R_d,R_d) 
    Wxh = np.random.randn(R_d,R_D)
    Why = np.random.randn(yt_row,R_d)
    bh = np.random.randn(R_d,1)
    by = np.random.randn(yt_row,1)
    parameters = {"Wxh":Wxh,"Whh":Whh,"Why":Why,"bh":bh,"by":by}
    return parameters