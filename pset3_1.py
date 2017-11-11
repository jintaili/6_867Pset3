"""
Created on Thu Nov  2 13:28:08 2017

6867 Pset3_1

@author: shenhao
"""

import os
path = "/Volumes/Transcend/Dropbox (MIT)/2017 Fall/6.867/psets/pset3"
data_path = path + '/hw3_resources/data'
os.chdir(path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn; seaborn.set()


'''1 ReLU and softmax''' 
os.chdir(data_path)
df = pd.read_csv('data_3class.csv', sep = ' ', header = None) # 
df.columns = ['x1', 'x2', 'y']


t = pd.get_dummies(df['y'])
df['x1']
df['x2']



def derivative_loss_wrt_a_L(y, a_L):
    '''
    This fun computes the derivative of y wrt a_L;
    a_L is the output of the last layer, which is the softmax function g()
    This fun involves one observation only. 
    y is a one-shot vector, (0, 0, 1, 0, 0, ... 0). In this application; K = 3, and (0, 1, 0) etc.

    loss = np.dot(y, np.log(a_L)), which is the cross-entropy loss in softmax; check Bishop 5.24
    derivative: d loss / d aL = y / a_L             
    '''
    if len(y) != len(a_L):
        raise NameError('Length of the inputs does not match')
    return y/a_L
    
## test
#y = np.array([[1,1,0]]).T; a_L = np.array([[0.3, 0.2, 0.5]]).T
#print(derivative_loss_wrt_a_L(y, a_L))
#y = np.array([[1,1,0,0.5]]).T; a_L = np.array([[0.3, 0.2, 0.5]]).T
#print(derivative_loss_wrt_a_L(y, a_L))

def softmax(z_L):
    '''
    Output is the softmax transformation of z_L
    '''
    a_L = np.exp(z_L) / np.sum(np.exp(z_L))
    return a_L

## test
#z_L = np.array([[1, 3, 4, 2, 5]]).T
#print(softmax(z_L))    


def derivative_matrix_softmax(z_L):
    '''
    This computes da_L/dz_L
    a_L = softmax(z_L)
    derivative = a_iL(1 - a_jL), if i = j; 
    derivative = - a_iL * a_jL, if i != j; 
    Use the full Jacobian matrix; This is used only once because softmax is the last layer
    Jacobian[i, j] = da_Li / dz_Lj
    
    output M:
     -- len(a_L) ---
    |
    |
    len(z_L)
    |
    |       
    
    '''
    a_L = softmax(z_L)
    derivative_matrix = np.zeros((len(a_L), len(z_L)))
    for i in range(len(a_L)): # rows
        for j in range(len(z_L)): # columns
            if i == j:
                derivative_matrix[i, j] = - a_L[i, 0] * a_L[j, 0] 
                # it does not matter i or j because i == j
            else:
                derivative_matrix[i, j] = a_L[i, 0] * (1 - a_L[j, 0]) 
    # return transpose so that this d(softmax_matrix) * (dl/da_L) dimensions match
    return derivative_matrix.T 
    
## test
#z_L = np.array([[1, 3, 4, 2]]).T              
#print(derivative_matrix_softmax(z_L))


def derivative_delta_L(y, z_L):
    '''
    Derivative of loss wrt z_L
    d loss / d z_L = (d loss / d a_L) * (d a_L / d z_L)
    '''
    a_L = softmax(z_L)
    dloss_daL = derivative_loss_wrt_a_L(y, a_L)
    daL_dzL = derivative_matrix_softmax(z_L)
    
    derivative_delta_L = np.dot(daL_dzL, dloss_daL) 
    return derivative_delta_L # (K * 1)
  
## test    
#z_L = np.array([[1, 2, 3, 4]]).T    
#y = np.array([[0, 0, 1, 0]]).T
#derivative_delta_L(y, z_L)


def derivative_loss_W_L(a_L_1, y, W_L, b_L):
    '''
    Inputs:
        a_L_1: output of the layer prior to the final layer (M * 1)
        y: choice (K * 1)
        W_L: parameter matrix (K * M)
        b_L: parameter vector (K * 1)
    Output:
        d loss / d W_L = (d loss / d z_L) * (d z_L / d W_L); 
    '''
    z_L = np.dot(W_L, a_L_1) + b_L
    
    # compute two derivatives
    derivative_loss_z_L = derivative_delta_L(y, z_L) # d loss / d z_L; (K * 1)
    dz_L_dW_L = a_L_1 # (M * 1)
                    
    # the final derivative 
    final_derivative = np.dot(derivative_loss_z_L, dz_L_dW_L.T)
    return final_derivative
    
## test
#W_L = np.array([[1, 0, 0, 0], 
#                [0, 1, 0, 0],
#                [0, 0, 1, 0]])
#b_L = np.array([[1, 1, 2]]).T
#a_L_1 = np.array([[1, 1, 1, 1]]).T
#y = np.array([[0, 0, 1]]).T
#print(derivative_loss_W_L(a_L_1, y, W_L, b_L))


def derivative_loss_b_L(a_L_1, y, W_L, b_L):
    '''
    Inputs:
        a_L_1: output of the layer prior to the final layer (M * 1)
        y: choice (K * 1)
        W_L: parameter matrix (K * M)
        b_L: parameter vector (K * 1)
    Output:
        d loss / d b_L = (d loss / d z_L) * (d z_L / d b_L) = (d loss / d z_L); 
        (K * 1)
    '''
    z_L = np.dot(W_L, a_L_1) + b_L    
    final_derivative = derivative_delta_L(y, z_L)
    return final_derivative
    
## test
#W_L = np.array([[1, 0, 0, 0], 
#                [0, 1, 0, 0],
#                [0, 0, 1, 0]])
#b_L = np.array([[1, 1, 2]]).T
#a_L_1 = np.array([[1, 1, 1, 1]]).T
#y = np.array([[0, 0, 1]]).T
#print(derivative_loss_b_L(a_L_1, y, W_L, b_L))


'''
Simulate the feed forward network
'''

def simulation_feed_forward_network(W_L, b_L):
    '''
    Simulate the data generation process of a softmax transformation
    Inputs: 
        parameters WL, bL
        dataset: a_L_1_matrix
    Output:
        y_matrix: choices
        a_L_1_matrix: input dataset, generated within this function    
    '''
    a_L_1_matrix = np.random.multivariate_normal(np.array([0, 0, 0, 0]), 
                                             np.array([[1, .5, .5, .5],
                                                       [.5, 1, .5, .5],
                                                       [.5, .5, 1, .5],
                                                       [.5, .5, .5, 1]]), size = 100) 
    y_matrix = np.zeros((100, 3))
    for i in range(a_L_1_matrix.shape[0]):
        z_L = np.dot(W_L, a_L_1_matrix[i, :].reshape(4, 1)) + b_L
        probability = softmax(z_L)
        y_matrix[i, np.argmax(probability)] = 1          
    
    return a_L_1_matrix, y_matrix

## test
#W_L = np.array([[1, 0, 0, 0], 
#                [0, 1, 0, 0],
#                [0, 0, 1, 0]])
#b_L = np.array([[1, 1, 2]]).T
#a_L_1_matrix, y_matrix = simulation_feed_forward_network(W_L, b_L)
#print('The output of the layer previous to the last layer (softmax) is: ' ,a_L_1_matrix)
#print('-----------------------')
#print('The output choice matrix is: ', y_matrix)


def learning_rate_SGD(t, tau0 = 1, kap = 0.75):
    '''
    This computes appropriate learning rates of SGD
    '''
    return (tau0 + t)**(-kap)
    
## test
#t_array = np.arange(100)
#rate_array = learning_rate_SGD(t_array)
#plt.plot(t_array, rate_array)


def estimate_feed_forward_network(y_matrix, a_L_1_matrix):
    '''
    This is a sanity test: given Y (y_matrix) and X (a_L_1_matrix), whether we could do estimation
    Estimate W_L and b_L by using SGD
    Use SGD
    '''



W_L = np.array([[1, 0, 0, 0], 
                [0, 1, 0, 0],
                [0, 0, 1, 0]])
b_L = np.array([[1, 1, 2]]).T
a_L_1_matrix, y_matrix = simulation_feed_forward_network(W_L, b_L)


N, M = a_L_1_matrix.shape
K = y_matrix.shape[1]

# initialization
W_initial = np.random.normal(loc = 0.0, scale = 1/M, size = (3, 4))
b_initial = np.zeros((K, 1))    

# stochasticity
T = 10000
for t in range(T):
    index = np.random.choice(N)
    y_random_t = y_matrix[index, :][:, np.newaxis]
    a_L_1_t = a_L_1_matrix[index, :][:, np.newaxis]
    
    # compute derivatives
    learning_rate = learning_rate_SGD(t)
    derivative_loss_W_L_t = derivative_loss_W_L(a_L_1_t, y_random_t, W_initial, b_initial)
    derivative_loss_b_L_t = derivative_loss_b_L(a_L_1_t, y_random_t, W_initial, b_initial)

    W_initial = W_initial - learning_rate * derivative_loss_W_L_t 
    b_initial = b_initial - learning_rate * derivative_loss_b_L_t 

print('The current W matrix is: ', np.round_(W_initial, decimals = 2))
print('The current b vector is: ', np.round_(b_initial, decimals = 2))
print('------------------------')



def cross_entropy_cost(y_matrix, a_L_1_matrix):
    '''
    This function computes the cross entropy cost of 
    
    '''






# Here plus because we want to maximize the target cross-entropy

        



    print(derivative_loss_W_L(a_L_1_t, y_random_t, W_initial, b_initial))
    print(derivative_loss_b_L(a_L_1_t, y_random_t, W_initial, b_initial))

    
    
    
# gradient descent
    
    





              
a_L_1 = np.array([[1, 1, 1, 1]]).T
z_L = np.dot(W_L, a_L_1) + b_L              
y = np.array([[0, 0, 1]]).T

            
            
            
            
            

              




'''
Suppose we have some z_L, then we could use the functions above to derive 

'''


def relu_activation(M):
    '''
    This is the relu activation function
    '''
    return M * (M > 0)

## test
#M = np.array([[-1, 1], [0, 0.1]])
#print(relu_activation(M))
    
def softmax_activation(z):
    '''
    This is the softmax activation function
    z is supposed to be one vector, output is the prob function
    '''
    prob_z = np.exp(z)/np.sum(np.exp(z)) 
    return prob_z
    
## test
#z = np.array([[0, -1, 1]]).T
#print(softmax_activation(z))


def objective_function_cross_entropy():
    ''' 
    This function is the cost of the optimization problem
    '''
    # For the first step, let's model 3 nodes latent layer    
    
    
    
    return np.nan    
    

def derivative_function_cross_entropy():
    ''' 
    This function is the derivative of the optimization problem
    '''
    return np.nan    
    

def optimize_function_cross_entropy():
    '''
    This function optimizes the cross entropy cost by using SGD
    '''
    return np.nan    










#2 ReLU and softmax




