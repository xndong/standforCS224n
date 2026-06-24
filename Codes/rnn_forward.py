# -*- coding: utf-8 -*-
"""
Created on Wed July 25 21:39:34 2019

@author: DongXiaoning
"""

import numpy as np


def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum(axis=0)
   
def sigmoid(x):       # scalar  
    return 1 / (1 + np.exp(-x))

def rnn_cell_forward(xt, h_prev, parameters):#    note that imput x can be vector(eg. word embedding) as well as matrix(eg speech features).
    """
    Arguments:
    xt -- your input data at timestep "t", numpy array of shape (n_x, m).    
       -- note that imput x can be vector(eg. word embedding) as well as matrix(eg speech features).  _dxn
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

def rnn_forward(x, h0, parameters):    
    """
    Arguments:
    x -- Input data for every time-step, of shape (n_x, m, T_x).
    h0 -- Initial hidden state, of shape (n_h, m)
    parameters -- python dictionary containing:
                        Wxh -- Weight matrix multiplying the input, numpy array of shape (n_h, n_x)
                        Whh -- Weight matrix multiplying the hidden state, numpy array of shape (n_h, n_h)
                        Why -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_h)
                        bh --  Bias numpy array of shape (n_h, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)

    Returns:
    h -- Hidden states for every time-step, numpy array of shape (n_h, m, T_x)
    y_pred -- Predictions for every time-step, numpy array of shape (n_y, m, T_x)
    caches -- tuple of values needed for the backward pass, contains (list of caches, x)
    """
    
    # Initialize "caches" which will contain the list of all caches
    caches = []    
    # Retrieve dimensions from shapes of x and parameters["Why"]
    n_x, m, T_x = x.shape
    n_y, n_h = parameters["Why"].shape   
    
    # initialize 'returns' h and y with zeros
    h = np.zeros((n_h, m, T_x))
    y_pred = np.zeros((n_y, m, T_x))    
    # Initialize h_next
    h_next = h0   
    
    # loop over all time-steps
    for t in range(T_x):        
        # Update next hidden state, compute the prediction, get the cache 
        h_next, yt_pred, cache = rnn_cell_forward(x[:,:,t], h_next, parameters)        
        # Save the value of the new "next" hidden state in h 
        h[:,:,t] = h_next
        # Save the value of the prediction in y
        y_pred[:,:,t] = yt_pred
        # Append "cache" to "caches" 
        caches.append(cache)    
        
    # store values needed for backward propagation in cache
    caches = (caches, x)
    return h, y_pred, caches


def main():
    np.random.seed(1)
    # xt = np.random.randn(10,8)
    # h_prev = np.random.randn(5,8)
    # Whh = np.random.randn(5,5) 
    # Wxh = np.random.randn(5,10)
    # Why = np.random.randn(20,5)  # 20-class 'classifier'
    # bh = np.random.randn(5,8)
    # by = np.random.randn(20,8)   # 20-class 'classifier'
    # parameters = {"Wxh":Wxh,"Whh":Whh,"Why":Why,"bh":bh,"by":by}
    # h_next,yt_pred,cache = rnn_cell_forward(xt,h_prev,parameters)   
    # print(h_next)
    # print(yt_pred)    # 20 * 8 shape
    
    x = np.random.randn(15,20,300)
    h0 = np.random.randn(5,20)
    Whh = np.random.randn(5,5)
    Wxh = np.random.randn(5,15)
    Why = np.random.randn(10,5) # 10-class 'classifier'
    bh = np.random.randn(5,20)
    by = np.random.randn(10,20)
    parameters =  {"Wxh":Wxh,"Whh":Whh,"Why":Why,"bh":bh,"by":by}
    h, y_pred, caches = rnn_forward(x, h0, parameters)
    print(y_pred)

if __name__ == "__main__":
    main()