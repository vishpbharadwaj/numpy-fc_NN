# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 08:28:20 2017

Description: Neural Network Activation function templete

@author: vishnu
"""

import numpy as np

def sigmoid(x):
    """
    Arguments:
        x -- A scalar or numpy array
    Return:
        s -- computed sigmoid function
    """
    ###################################
    s = 1/(1+np.exp(-x))
    ###################################
    return s
" --------- end of function ------------------------------------------------- "


def sigmoid_derivative(x):
    """
    Arguments:
    x -- A scalar or numpy array
    Return:
    ds --  computed gradient.
    """
    ###################################
    s = sigmoid(x)
    ds = s*(1-s)
    ###################################
    return ds
" --------- end of function ------------------------------------------------- "



def tanh(x):
    """
    Arguments:
        x -- A scalar or numpy array
    Return:
        s -- computed tanh function
    """
    ###################################
    s = np.tanh(x)
    ###################################
    return s
" --------- end of function ------------------------------------------------- "

def tanh_derivative(x):
    """
    Arguments:
        x -- A scalar or numpy array
    Return:
        s -- computed gradient of tanh
    """
    ###################################
    s = 1 - (np.tanh(x) ** 2)
    ###################################
    return s
" --------- end of function ------------------------------------------------- "

def actv_gradient(x):
    """
    Arguments:
        x -- A scalar or numpy array
    Return:
        dx -- computed gradient of custom function
    """
    ###################################
    dx = tanh_derivative(x)
    ###################################
    return dx
" --------- end of function ------------------------------------------------- "