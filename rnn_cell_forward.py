# -*- coding: utf-8 -*-
"""
Created on Wed July 25 21:39:34 2019

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

def rnn_cell_forward(xt, h_prev, parameters):    
    """
    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).
    h_prev -- Hidden state at timestep "t-1", numpy array of shape (n_h, m)
    parameters -- python dictionary containing:
                        Wxh -- Weight matrix multiplying the input, numpy array of shape (n_h, n_x)
                        Whh -- Weight matrix multiplying the hidden state, numpy array of shape (n_h, n_h)
                        Why -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_h)
                        bh --  Bias, numpy array of shape (n_h, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    Returns:
    h_next -- next hidden state, of shape (n_h, m)
    yt_pred -- prediction at timestep "t", numpy array of shape (n_y, m)
    cache -- tuple of values needed for the backward pass, contains (h_next, h_prev, xt, parameters)
    """

    # Retrieve parameters from "parameters"
    Wxh = parameters["Wxh"]
    Whh = parameters["Whh"]
    Why = parameters["Why"]
    bh = parameters["bh"]
    by = parameters["by"]
    # compute next activation state using the formula given above
    h_next = np.tanh(np.matmul(Wxh, xt) + np.matmul(Whh, h_prev) + bh)   # + bh ---> boradcast广播机制  
    # compute output of the current cell using the formula given above
    yt_pred = softmax(np.matmul(Why, h_next) + by) 

    # store values you need for backward propagation in cache
    cache = (h_next, h_prev, xt, parameters)    
    return h_next, yt_pred, cache

def main():
    np.random.seed(1)
    xt = np.random.randn(10,1)
    h_prev = np.random.randn(5,1)
    Whh = np.random.randn(5,5) 
    Wxh = np.random.randn(5,10)
    Why = np.random.randn(100,5)  # 100-class 'classifier'
    bh = np.random.randn(5,1)
    by = np.random.randn(100,1)   # 100-class 'classifier'
    parameters = {"Wxh":Wxh,"Whh":Whh,"Why":Why,"bh":bh,"by":by}
    h_next,yt_pred,cache = rnn_cell_forward(xt,h_prev,parameters)
    print(h_next)
    print(yt_pred)

if __name__ == "__main__":
    main()