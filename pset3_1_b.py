import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

data_path = r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw3\data'
os.chdir(data_path)
df = pd.read_csv('data_3class.csv', sep=' ', header=None)  #
df.columns = ['x1', 'x2', 'y']
X = np.asarray(df.iloc[:,0:2])
y = np.asarray(df.iloc[:,2].astype(int))

num_examples = len(df)  # training set size
nn_input_dim = 2  # input layer dimensionality
nn_output_dim = 3  # output layer dimensionality

# Gradient descent parameters (I picked these by hand)
epsilon = 0.01  # learning rate for gradient descent
reg_lambda = 0.01  # regularization strength

def softmax(z_L):
    '''
    Output is the softmax transformation of z_L
    '''
    a_L = np.exp(z_L) / np.sum(np.exp(z_L), axis=1, keepdims=True)
    return a_L

def learning_rate_SGD(t, tau0=1, kap=0.75):
    '''
    This computes appropriate learning rates of SGD
    '''
    return (tau0 + t) ** (-kap)

def relu_activation(M):
    '''
    This is the relu activation function
    '''
    return M * (M > 0)
def tanh_activation(M):
    '''
    This is the relu activation function
    '''
    return np.tanh(M)

# Helper function to evaluate the total loss on the dataset
def calculate_error(model, X, y):
    return null

def calculate_loss_batch(model, X, y):
    num_examples = np.size(X, 0)
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = X.dot(W1) + b1
    a1 = tanh_activation(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return 1./num_examples * data_loss

def calculate_loss_stoch(model, x_pt, y_pt):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = x_pt.dot(W1) + b1
    a1 = tanh_activation(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[:,y_pt])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
    return data_loss

# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, X, y, num_passes=20000, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = np.size(X, 0)
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        # Forward propagation
        z1 = X.dot(W1) + np.repeat(b1, num_examples, 0)
        a1 = tanh_activation(z1)
        z2 = a1.dot(W2) + np.repeat(b2, num_examples, 0)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0)

        # Add regularization terms (b1 and b2 don't have regularization terms)
        dW2 += reg_lambda * W2
        dW1 += reg_lambda * W1

        # Gradient descent parameter update
        epsilon = 0.01
        W1 += -epsilon * dW1
        b1 += -epsilon * db1
        W2 += -epsilon * dW2
        b2 += -epsilon * db2

        # Assign new parameters to the model
        model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 1000 == 0:
            print(
            "Loss after iteration %i: %f" % (i, calculate_loss_batch(model, X, y))
            )

    return model

def build_model_SGD(nn_hdim, X, y, num_passes=500, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    num_examples = np.size(X, 0)
    loss_T = []
    np.random.seed(0)
    W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hdim))
    W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
    b2 = np.zeros((1, nn_output_dim))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i in range(0, num_passes):

        indices_examples = range(num_examples)

        for j in range(num_examples):

            pt = np.random.choice(indices_examples,replace=False)
            x_pt = np.reshape(X[pt,:],(1,-1))
            y_pt = y[pt]

            # Forward propagation
            z1 = x_pt.dot(W1) + b1
            a1 = tanh_activation(z1)
            z2 = a1.dot(W2) + b2
            exp_scores = np.exp(z2)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation
            delta3 = np.copy(probs)
            delta3[0,y_pt] -= 1
            dW2 = (a1.T).dot(delta3)
            db2 = np.sum(delta3, axis=0, keepdims=True)
            delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
            dW1 = np.dot(x_pt.T, delta2)
            db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            dW2 += reg_lambda * W2
            dW1 += reg_lambda * W1

            # Gradient descent parameter update
            epsilon = learning_rate_SGD((i*num_examples + j+1)/2000, tau0 = 0.3, kap = 0.55)
            W1 += -epsilon * dW1
            b1 += -epsilon * db1
            W2 += -epsilon * dW2
            b2 += -epsilon * db2

            # Assign new parameters to the model
            model = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss and i % 10 == 0:
            loss_T.append(calculate_loss_batch(model, X, y))
            print(
            "Loss after epoch %i: %f" % (i, loss_T[-1])
            )

    return (model,loss_T)


plt.plot(build_model_SGD(3, X, y, print_loss=True)[1])