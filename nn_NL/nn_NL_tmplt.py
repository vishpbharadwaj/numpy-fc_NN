# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 15:36:17 2017

Description: Neural Network for N-Layer templete

@author: vishnu
"""

import numpy as np
import nn_actv_tmplt as nnactv

"""
for this templete to work, the data should be in the form of
        | x1 x1+1  . x1+1 |
        | x2 x2+1  . x2+1 |
    X = | .   .    .   .  |
        | .   .    .   .  |
        | xn xn+1  . xk+1 |

and Y will be
        | y1 y1+1  . y1+1 |
        | y2 y2+1  . y2+1 |
    Y = | .   .    .   .  |
        | .   .    .   .  |
        | yn yn+1  . yk+1 |
"""

#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
def initialize_parameters(inpt_len, nHL, outp_len):
    """
    Description:
        initializes each value of weights according to the configuratin
        with random numbers and bias with zeros
        
    Argument:
        inpt_len -- size of the input layer
        nHL -- No. of hidden layers
        outp_len -- size of the output layer
        
    Returns:
        hiddlay_size -- array of size(length) of each hidden layer
        W_params -- python dict - weight matrix of shape (n_h, n_x)
        B_params -- python dict - bias vector of shape (n_h, 1)
            
    Notes: As of now, the length or size of each hidden layer is fixed in the 
        first for loop, this has to be changed manually. Later dynamic models 
        has to be added. TODO: 
    """
    ###################################  
    hiddlay_size = np.ones([1,nHL],dtype=int)
    W_params = {}
    B_params = {}

    for i in range(nHL):
        #initializing to 4 neuron in each hidden layer
        hiddlay_size[0,i] = 4
    " ---- for loop ends --- "
    
    """ Notes for TODO
    for 1st layer w0 and bo and 2nd w1 and b1 and so on ...

    using 3D matrix for both weight and bias as we have the flexibility of calling
    in dict we have to know the w(i) value where i is not controllable 
    but this feature is not suitable now as 3-D matrix should be in cube shape which
    will pose a problem if the hidden layers of different length.

    can use array of arrays for now and later if feasible can shift to dict.
    """
    for i in range(nHL+1):# nHL+1 because No. of weight matrices required for whole NN is hidden layer +1
        np.random.seed(1)
        if (i == 0):
            W_params["w{0}".format(i)] = np.random.randn(hiddlay_size[0,0],inpt_len)
            B_params["b{0}".format(i)] = np.zeros([hiddlay_size[0,0],1])
        elif (i < hiddlay_size.size ):
            W_params["w{0}".format(i)] = np.random.randn(hiddlay_size[0,i],hiddlay_size[0,i-1])
            B_params["b{0}".format(i)] = np.zeros([hiddlay_size[0,i],1])
        else:
            W_params["w{0}".format(i)] = np.random.randn(outp_len,hiddlay_size[0,i-1])
            B_params["b{0}".format(i)] = np.zeros([outp_len,1])
    " -------- for loop ends -------- "
    ###################################
    return hiddlay_size, W_params, B_params
" --------- end of function -------------------------------------------------- "
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH


#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
def activation_fn(z_inactv):
    """
    Arguments:
        z_inactv -- A scalar or numpy array
        
    Return:
        z_actv -- computed appropriate activation function
        
    Notes:
        can compute different activation function depending, should be changed once here
    """
    ###################################
    #z_actv = nnactv.sigmoid(z_inactv)
    z_actv = nnactv.tanh(z_inactv)
    ###################################
    return z_actv
" --------- end of function -------------------------------------------------- "
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH


#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
def forward_propagation(inpt, W_params, B_params):
    """
    Description:
        forward propagation of the net with inputs
        
    Argument:
        inpt -- input data of size (n_x, m)
        W_params -- python dict containing Weight params (output of initialization function)
        B_params -- python dict containing Bias params (output of initialization function)
    
    Returns:
        Z_cache -- python dict containing Zs of each layer
        A_cache -- python dict containing As of each layer
    """
    ###################################
    Z_cache = {}
    A_cache = {}
    for i in range(len(W_params)):
        if (i == 0):
            Z_cache["z{0}".format(i)] = np.dot( (W_params["w{0}".format(i)]), inpt ) + B_params["b{0}".format(i)]
            A_cache["a{0}".format(i)] = activation_fn(Z_cache["z{0}".format(i)])
        else:
            Z_cache["z{0}".format(i)] = np.dot( (W_params["w{0}".format(i)]), A_cache["a{0}".format(i-1)] ) + B_params["b{0}".format(i)]
            A_cache["a{0}".format(i)] = activation_fn(Z_cache["z{0}".format(i)])
    " -------- for loop ends -------- "
    ###################################
    return Z_cache, A_cache
" --------- end of function -------------------------------------------------- "
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH


#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
def cross_entropy_cost(An, org_out):
    """
    Description:
        Computes the cost using custom function
    
    Arguments:
        An -- The activation output (from last layer i.e. output layer) 
                    with shape (1, number of examples) for one output
        org_out -- "true" labels vector of shape (1, number of examples) for one output
    
    Returns:
        cost -- cross-entropy cost
    """
    ###################################
    logprobs = np.multiply(np.log(An), org_out) + np.multiply((1 - org_out), np.log(1 - An))
    cost = -np.sum(logprobs)    
    cost = np.squeeze(cost)     # makes sure cost is the dimension we expect.E.g., turns [[x]] into x 
    assert(isinstance(cost, float))
    ###################################
    return cost
" --------- end of function ------------------------------------------------- "
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH


#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
def cross_entropy_gradient(An, org_out):
    """
    Description:
        Computes the gradient of cross_entropy loss
    
    Arguments:
        An -- The activation output (from last layer i.e. output layer) 
                    with shape (1, number of examples) for one output
        org_out -- "true" labels vector of shape (1, number of examples) for one output
    
    Returns:
        dA -- cross-entropy cost derivative
    """
    dA = 1;
    ###################################
    # this will pose a problem as An can be 1 and the raw 
    # derivation of Actv will have problem if (1-An) is zero
    #dA = (-org_out/An) + ((1-org_out)/(1-An))
    ###################################
    return dA
" -------- end of function -------------------------------------------------- "
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH


#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
def backward_propagation(inpt, org_out, W_params, B_params, Z_cache, A_cache):
    """
    Description:
        Implement the backward propagation.
    
    Arguments:
        inpt -- input data of size (n_x, m)
        org_out -- "true" labels vector of shape (1, number of examples) for one output
        W_params -- python dict containing Weight params (output of initialization function)
        B_params -- python dict containing Bias params (output of initialization function)
        Z_cache -- python dict containing Zs of each layer
        A_cache -- python dict containing As of each layer
    
    Returns:
        dW_params -- python dict containing weight gradients
        dB_params -- python dict containing bias gradients
    """
    ###################################
    dA_params = {}
    dZ_params = {}
    dW_params = {}
    dB_params = {}
    inpt_size = inpt.shape[1]
    
    lst_val = len(W_params) - 1
    
    for i in reversed(range(len(W_params))):# coming from last value to calculate gradient
        if (i == lst_val):
            #dA_params["da{0}".format(i)] = cross_entropy_gradient(A_cache["a{0}".format(i)],org_out)
            #dZ_params["dz{0}".format(i)] = dA_params["da{0}".format(i)] * nnactv.actv_gradient(A_cache["a{0}".format(i)])
            dZ_params["dz{0}".format(i)] = A_cache["a{0}".format(i)] - org_out
            dW_params["dw{0}".format(i)] = (1/inpt_size) * np.sum(A_cache["a{0}".format(i)] * dZ_params["dz{0}".format(i)],axis=1)
            dB_params["db{0}".format(i)] = (1/inpt_size) * np.sum(dZ_params["dz{0}".format(i)],axis=1)
        else:
            dA_params["da{0}".format(i)] = np.dot(W_params["w{0}".format(i+1)].T,dZ_params["dz{0}".format(i+1)])
            dZ_params["dz{0}".format(i)] = dA_params["da{0}".format(i)] * nnactv.actv_gradient(A_cache["a{0}".format(i)])
            dW_params["dw{0}".format(i)] = (inpt_size) * np.sum(A_cache["a{0}".format(i)] * dZ_params["dz{0}".format(i)],axis=1)
            dB_params["db{0}".format(i)] = (inpt_size) * np.sum(dZ_params["dz{0}".format(i)],axis=1)
        
    " ----- end of for loop ------------------------ "
    
    #column stacking because making the dimensions match for update_parameters
    for i in range(len(W_params)):# nHL+1 because No. of weight matrices required for whole NN is hidden layer +1
        dW_params["dw{0}".format(i)] = np.column_stack([dW_params["dw{0}".format(i)],])
        dB_params["db{0}".format(i)] = np.column_stack([dB_params["db{0}".format(i)],])
    " ----- end of for loop ------------------------ "
    ###################################
    return dW_params, dB_params
" --------- end of function ------------------------------------------------- "
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH


#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
def update_parameters(W_params, B_params, dW_params, dB_params,learning_rate, nHL):
    """
    Description:
        Updates parameters using the gradient descent update
    
    Arguments:
        W_params -- python dict containing weights
        B_params -- python dict containing bias
        dW_params -- python dict of weights
        dB_params -- python dict of bias
        learning_rate  -- rate of the descent
        nHL -- No. of hidden layers
    
    Returns:
        W_params -- python dict containing weights
        B_params -- python dict containing bias 
    """
    ###################################
    for i in range(nHL+1):# nHL+1 because No. of weight matrices required for whole NN is hidden layer +1
        W_params["w{0}".format(i)] = W_params["w{0}".format(i)] - (learning_rate) * (dW_params["dw{0}".format(i)])
        B_params["b{0}".format(i)] = B_params["b{0}".format(i)] - (learning_rate) * (dB_params["db{0}".format(i)])
    " ---- end of for loop ------ "
    ###################################
    return W_params, B_params
" --------- end of function ------------------------------------------------- "
#HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH