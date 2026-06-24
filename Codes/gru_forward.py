# -*- coding: utf-8 -*-
"""
Created on Tue Nov 26 13:56:58 2019

@author: DongXiaoning
"""

import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  
    return e_x / e_x.sum(axis=0)
   
def sigmoid(x):       
    return 1 / (1 + np.exp(-x))

def gru_cell_forward(xt, h_prev, parameters):
    '''

    Parameters
    ----------
    xt : ndarray
        has the (n_x, m) shape. n_x is defined by your data format.  
    h_prev : ndarray
        has the (n_h, m) shape. n_h, I think, is defined by you.
    parameters : Dictionary, W,U,Wr...Uz notations come from stanford cs224 lecture note and origin paper
        W:  weight matrix from xt to ht_tilde, and has the shape (n_h, n_x)
        U:  weight matrix from h_prev to ht_tilde, and has the shape (n_h, n_h)
        Wr: weight matrix from xt to rt
        Ur: weight matrix from h_prev to rt
        Wz: weight matrix from xt to zt
        Uz: weight matrix from h_prev to zt
        Why: weight matrix from hidden state to output y, and has the shape (n_y, n_h). n_y is defined by your task eg. 10-class classification.
        bh: bias for hidden state, has the shape (n_y, 1)
        by: bias for output y, has the shape (n_h, 1)
    
    Returns
    -------
    return hidden state, output and cache(h_next, h_prev, xt, parameters)
    '''
    
    # retrieve W, U, Wr, Ur,...,by and bh from parameters.
    W = parameters['W']
    U = parameters['U']
    Wr = parameters['Wr']
    Ur = parameters['Ur']
    Wz = parameters['Wz']
    Uz = parameters['Uz']
    Why = parameters['Why']
    bh = parameters['bh']
    by = parameters['by']
    
    zt = sigmoid(np.matmul(Wz, xt) + np.matmul(Uz, h_prev))
    rt = sigmoid(np.matmul(Wr, xt) + np.matmul(Ur, h_prev))                             # np.matmul   ---> matrix product
    ht_tilde = np.tanh(np.matmul(W, xt) + np.multiply(rt, np.matmul(U, h_prev)))        # np.multiply ---> hardamard product
    ht = np.tanh(np.multiply((1 - zt), ht_tilde) + np.multiply(zt, h_prev) + bh)        # 1 - zt  ---> broadcast mechanism
    yt = softmax(np.matmul(Why, ht) + by)
    cache = (h_prev, ht, yt, parameters)
   
    return ht, yt, cache

def gru_forward(x, h0, parameters):
    '''
    

    Parameters
    ----------
    x : ndarray
        input matrix, (n_x, m, T_x). T_x is how many entries data.
    h0 : initial hidden state h0
        hidden state matrix, (n_h, m)
    parameters : Dictionary
        W:  weight matrix from xt to ht_tilde, and has the shape (n_h, n_x)
        U:  weight matrix from h_prev to ht_tilde, and has the shape (n_h, n_h)
        Wr: weight matrix from xt to rt
        Ur: weight matrix from h_prev to rt
        Wz: weight matrix from xt to zt
        Uz: weight matrix from h_prev to zt
        Why: weight matrix from hidden state to output y, and has the shape (n_y, n_h). n_y is defined by your task eg. 10-class classification.
        bh: bias for hidden state, has the shape (n_y, 1)
        by: bias for output y, has the shape (n_h, 1)
        
    Returns
    -------
    h : ndarray
        hidden states for each entry of data, (n_h, m, T_x)
    y : ndarray
        outputs for each entry of data, (n_y, m, T_x)
    caches : List
        cache for every iteration in T_x
    '''
    
    # retrieve dimensions
    n_x, m, T_x = x.shape
    n_y, n_h = parameters['Why'].shape
    
    # initialize 'returns' with zeros or empty
    h = np.zeros((n_h, m, T_x))
    y = np.zeros((n_y, m, T_x))
    caches = []
    
    ht = h0
    for i in range(T_x):
        ht, yt, cache = gru_cell_forward(x[:,:,i], ht, parameters)
        h[:,:,i] = ht
        y[:,:,i] = yt
        caches.append(cache)
    
    caches = (caches, x)
    
    return h, y, caches



    
np.random.seed(1)
# xt = np.random.randn(200,1)
# h_prev = np.random.randn(50,1)
# W = np.random.randn(50,200)
# U = np.random.randn(50,50)
# Wr = np.random.randn(50,200)
# Ur = np.random.randn(50,50)
# Wz = np.random.randn(50,200)
# Uz = np.random.randn(50,50)
# Why = np.random.randn(8,50)  # 8-class 'classifier'
# bh = np.random.randn(50,1)
# by = np.random.randn(8,1)   # 8-class 'classifier'
# parameters = {'W':W,'U':U,'Wr':Wr,'Ur':Ur,'Wz':Wz,'Uz':Uz,'Why':Why,'bh':bh,'by':by}

# h_next,yt_pred,cache = gru_cell_forward(xt, h_prev, parameters)   



x = np.random.rand(200,1,1000)
h0 = np.random.randn(50,1)
W = np.random.randn(50,200)
U = np.random.randn(50,50)
Wr = np.random.randn(50,200)
Ur = np.random.randn(50,50)
Wz = np.random.randn(50,200)
Uz = np.random.randn(50,50)
Why = np.random.randn(8,50)  # 8-class 'classifier'
bh = np.random.randn(50,1)
by = np.random.randn(8,1)   # 8-class 'classifier'
parameters = {'W':W,'U':U,'Wr':Wr,'Ur':Ur,'Wz':Wz,'Uz':Uz,'Why':Why,'bh':bh,'by':by}

h, y, caches = gru_forward(x, h0, parameters) 
    
    
    
    
    
    