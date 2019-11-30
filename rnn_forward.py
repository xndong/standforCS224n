# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:54:35 2019

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

# n : n
def rnn_forward_nn(xt,h_prev,parameters):  
    Wxh = parameters["Wxh"]
    Whh = parameters["Whh"]
    Why = parameters["Why"]
    bh = parameters["bh"]
    by = parameters["by"]
    xt_row,xt_col = np.shape(xt)
    for i in range(xt_row):
        h_next = np.tanh(np.matmul(Wxh, xt[i,:].reshape(xt_col,1)) + np.matmul(Whh, h_prev) + bh)
        yt_pred = softmax(np.matmul(Why, h_next) + by) 
        # print(h_next)
        h_prev = h_next
        print(yt_pred)    
    return h_next, yt_pred

# n : 1
def rnn_forward_n1(xt,h_prev,parameters):  
    Wxh = parameters["Wxh"]
    Whh = parameters["Whh"]
    Why = parameters["Why"]
    bh = parameters["bh"]
    by = parameters["by"]
    xt_row,xt_col = np.shape(xt)
    for i in range(xt_row):
        h_next = np.tanh(np.matmul(Wxh, xt[i,:].reshape(xt_col,1)) + np.matmul(Whh, h_prev) + bh)
        # print(h_next)
        h_prev = h_next
    yt_pred = softmax(np.matmul(Why, h_next) + by) 
    print(yt_pred)    
    return h_next, yt_pred

# 1 : n
def rnn_forward_1n(xt,h_prev,parameters):  
    Wxh = parameters["Wxh"]
    Whh = parameters["Whh"]
    Why = parameters["Why"]
    bh = parameters["bh"]
    by = parameters["by"]
    xt_row,xt_col = np.shape(xt)
    h_next = np.tanh(np.matmul(Wxh, xt[0,:].reshape(xt_col,1)) + np.matmul(Whh, h_prev) + bh)
    yt_pred = softmax(np.matmul(Why, h_next) + by) 
    print(yt_pred)    
    for i in range(10):   # 1 ： 10
        h_prev = h_next
        h_next = np.tanh(np.matmul(Whh, h_prev) + bh)
        yt_pred = softmax(np.matmul(Why, h_next) + by) 
        # print(h_next)
        print(yt_pred)    
    return h_next, yt_pred

# n : m a.k.a. seq2seq model
    
def main():
    np.random.seed(1)
    xt = np.random.randn(60,1000)
    h_prev = np.random.randn(800,1)
    
    _,R_D = np.shape(xt)
    R_d,_ = np.shape(h_prev)
    yt_row = 10 # 10-class 'classifier'
    parameters = init_parameter(R_D,R_d,yt_row)

    rnn_forward_1n(xt,h_prev,parameters)    
    
    
if __name__ == "__main__":
    main()