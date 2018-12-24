# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 19:42:45 2017

Description: Basics implementation of N-Layer Neural Network

@author: vishnu
"""

import numpy as np
import matplotlib.pyplot as plt
import pylab as pl
import h5py as h5py

import sys
sys.path.insert(0,'C:\\Users\\vishnubh\\Documents\\work\\source_code\\python\\nn_templete\\nn_NL')
import nn_NL_tmplt as nnNL


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

i.e. X  = [No. of Input nodes  ,  No. of datasets to each node] and 
     Y  = [1 , successive values of output] -- this is for one output

"""


"""
Load the data and set the sizes for data
"""
################################################################################

def load_data_from_text_3col(filename):
    first_col, second_col, third_col = np.loadtxt(filename, unpack=True, ndmin=2)
    return first_col, second_col, third_col

################################################################################

################################################################################
def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# Example of a picture
#index = 27
#plt.imshow(train_set_x_orig[index])
#print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0],-1).T

train_set_y = train_set_y
test_set_y = test_set_y

train_set_x_flatten = train_set_x_flatten/255
test_set_x_flatten = test_set_x_flatten/255
################################################################################

################################################################################
inpt_len = train_set_x_flatten.shape[0]
outp_len = 1
nHL = 3 #No. of hidden layers in NN

x1, x2, y_org = load_data_from_text_3col('logistic_regression_1.txt')
inpt = np.row_stack([x1.transpose(),x2.transpose()]) 
org_out = np.column_stack([y_org,]).transpose()

hiddlay_size, W_params, B_params = nnNL.initialize_parameters(inpt_len, nHL, outp_len)

iterations = 1000
learning_rate = 0.5

def nn_model(inpt,org_out,W_params,B_params,learning_rate,nHL,iterations):
    for i in range(iterations):
        Z_cache, A_cache = nnNL.forward_propagation(inpt,W_params,B_params) 
        
#        An = A_cache["a{0}".format(nHL)]
#        cost = nnNL.cross_entropy_cost(An,org_out)
    
        dW_params, dB_params = nnNL.backward_propagation(inpt,org_out,W_params,B_params,Z_cache,A_cache)
    
        W_params, B_params = nnNL.update_parameters(W_params,B_params,dW_params,dB_params,learning_rate,nHL)
    
    return W_params, B_params
" --------- end of function ----------------------- "
################################################################################


################################################################################
W_params, B_params = nn_model(train_set_x_flatten,train_set_y,W_params,B_params,learning_rate,nHL,iterations)
Z_cache, A_cache = nnNL.forward_propagation(test_set_x_flatten,W_params,B_params) 
An = A_cache["a{0}".format(nHL)]

print("predictions mean = " + str(np.mean(An)))
print ('Accuracy: %d' % float((np.dot(test_set_y,An.T) + np.dot(1-test_set_y,1-An.T))/float(test_set_y.size)*100) + '%')

################################################################################
# X - some data in 2dimensional np.array
#h = 0.5
#
#x1_min, x1_max = inpt[0, :].min() - 1, inpt[0, :].max() + 1
#x2_min, x2_max = inpt[1, :].min() - 1, inpt[1, :].max() + 1
#xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, h),
#                     np.arange(x2_min, x2_max, h))
#
#Xn = np.row_stack([xx1.ravel().transpose(), xx2.ravel().transpose()])
## here "model" is your model's prediction (classification) function
#Z_cache, A_cache = nnNL.forward_propagation(inpt,W_params,B_params) 
#Z = A_cache["a{0}".format(nHL)]
## Put the result into a color plot
#Z = Z.reshape(xx1.shape)
#
#fig2 = plt.figure()
#ax2 = fig2.add_subplot(111)
#
#ax2.contourf(xx1, xx2, Z)
#ax2.axis('on')
#
## Plot also the training points
#ax2.scatter(inpt.transpose()[:,0], inpt.transpose()[:,1], s=40, c=org_out.astype(int))
################################################################################