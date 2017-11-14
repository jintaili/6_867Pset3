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
converge_error = 1e-4

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

def predict(model, X):
    num_examples = np.size(X, 0)
    W_h, b_h = model['W_h'], model['b_h']
    nn_hl = len(W_h) - 1
    # Forward propagation to calculate our predictions
    a = X
    for i in range(nn_hl):
        z = a.dot(W_h[i]) + b_h[i]
        a = relu_activation(z)

    z = a.dot(W_h[-1]) + b_h[-1]
    exp_scores = np.exp(z)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probs

# Helper function to evaluate the total loss on the dataset
def calculate_error(model, X, y):
    num_examples = np.size(X, 0)
    W_h, b_h= model['W_h'], model['b_h']
    nn_hl = len(W_h) - 1
    # Forward propagation to calculate our predictions
    a = X
    for i in range(nn_hl):

        z = a.dot(W_h[i]) + b_h[i]
        a = relu_activation(z)

    z = a.dot(W_h[-1]) + b_h[-1]
    exp_scores = np.exp(z)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    predict = np.argmax(probs, 1)
    same = sum(predict == y) / num_examples
    return 1-same

def calculate_loss_batch(model, X, y, reg_lambda):
    num_examples = np.size(X, 0)
    W_h, b_h = model['W_h'], model['b_h']
    nn_hl = len(W_h) - 1
    # Forward propagation to calculate our predictions
    a = X
    for i in range(nn_hl):
        z = a.dot(W_h[i]) + b_h[i]
        a = relu_activation(z)

    z = a.dot(W_h[-1]) + b_h[-1]
    exp_scores = np.exp(z)
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    # Calculating the loss
    corect_logprobs = -np.log(probs[range(num_examples), y])
    data_loss = np.sum(corect_logprobs)
    # Add regulatization term to loss (optional)
    for i in range(len(W_h)):
        data_loss += reg_lambda/2 * (np.square(np.linalg.norm(W_h[i])))
    return 1./num_examples * data_loss

def calculate_loss_stoch(model, x_pt, y_pt):
    W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
    # Forward propagation to calculate our predictions
    z1 = x_pt.dot(W1) + b1
    a1 = relu_activation(z1)
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
        a1 = relu_activation(z1)
        z2 = a1.dot(W2) + np.repeat(b2, num_examples, 0)
        exp_scores = np.exp(z2)
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

        # Backpropagation
        delta3 = probs
        delta3[range(num_examples), y] -= 1
        dW2 = (a1.T).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3.dot(W2.T).dot(np.diag(np.reshape((z1>0)*1, -1)))
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
            "Loss after iteration %i: %f" % (i, calculate_loss_batch(model, X, y, reg_lambda))
            )

    return model

def build_model_SGD_tanh(nn_hdim, X, y, num_passes=500, print_loss=False):
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
            loss_T.append(calculate_loss_batch(model, X, y, reg_lambda))
            print(
                "Loss after epoch %i: %f" % (i, loss_T[-1])
            )
            print(
                "Error after epoch %i: %f" % (i, calculate_error(model, X, y))
            )

    return (model,loss_T)

# This function learns parameters for a multilayer neural network
# with ReLU activations and softmax output, and returns the model.
# - nn_hdim: A list length of the number of hidden layers,
#           each element stands for the number of nodes in each layer.
# - X and y: input
# - reg_lambda: Regularization coefficient.
# - converge_error: Stop criterion: absolute difference in consecutive run loss.
# - num_passes: Max number of Epochs through the training data for gradient descent
# - print_loss: If True, print the loss every 10 epochs
def build_model_SGD(nn_hdim, X, y, reg_lambda = 0.01, converge_error = 1e-4, epsilon = 0.01,
                    num_passes=150, print_loss=False):
    # Initialize the parameters to random values. We need to learn these.
    nn_hl = len(nn_hdim)
    num_examples = np.size(X, 0)
    nn_input_dim = np.size(X, 1)
    nn_output_dim = len(np.unique(y))
    loss_T = []
    # Initialization:
    np.random.seed(0)
    W_h = []
    b_h = []
    prev_dim = nn_input_dim
    for i in range(nn_hl):
        temp_hdim = nn_hdim[i]
        W_h.append(np.random.randn(prev_dim, temp_hdim) / np.sqrt(prev_dim))
        b_h.append(np.zeros((1, temp_hdim)))
        prev_dim = temp_hdim

    #W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
    #b1 = np.zeros((1, nn_hdim))

    #W_L = np.random.randn(nn_hdim[-1], nn_output_dim) / np.sqrt(nn_hdim[-1])
    #b_L = np.zeros((1, nn_output_dim))
    W_h.append(np.random.randn(nn_hdim[-1], nn_output_dim) / np.sqrt(nn_hdim[-1]))
    b_h.append(np.zeros((1, nn_output_dim)))

    # This is what we return at the end
    model = {}

    # Gradient descent. For each batch...
    for i_t in range(0, num_passes):

        indices_examples = range(num_examples)

        for j in range(num_examples):

            z_h = []
            a_h = []

            pt = np.random.choice(indices_examples,replace=False)
            x_pt = np.reshape(X[pt,:],(1,-1))
            y_pt = y[pt]

            # Forward propagation
            a_h.append(x_pt)

            for i in range(nn_hl):
                z_l = a_h[-1].dot(W_h[i]) + b_h[i]
                z_h.append(z_l)
                a_l = relu_activation(z_l)
                a_h.append(a_l)

            z_L = a_h[-1].dot(W_h[-1]) + b_h[-1]
            exp_scores = np.exp(z_L)
            probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # z1 = x_pt.dot(W1) + b1
            # a1 = relu_activation(z1)
            # z2 = a1.dot(W2) + b2
            # exp_scores = np.exp(z2)
            # probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

            # Backpropagation

            delta_L = np.copy(probs)
            delta_L[0,y_pt] -= 1
            # dW2 = (a_h[-1].T).dot(delta_L)
            # db2 = delta_L

            delta_l = delta_L
            dW_h = [None] * len(W_h)
            db_h = [None] * len(b_h)
            for j in range(len(W_h)):
                i = len(W_h) - j - 1
                dW_h[i] = np.dot(a_h[i].T, delta_l) + reg_lambda * W_h[i]
                db_h[i] = delta_l
                try:
                    delta_l = delta_l.dot(W_h[i].T).dot(np.diag(np.reshape((z_h[i-1]>0)*1,-1)))
                except:
                    pass

            # delta2 = delta_L.dot(W2.T).dot(np.diag(np.reshape((z1>0)*1,-1)))
            # dW1 = np.dot(x_pt.T, delta2)
            # db1 = np.sum(delta2, axis=0)

            # Add regularization terms (b1 and b2 don't have regularization terms)
            # dW2 += reg_lambda * W2
            # dW1 += reg_lambda * W1

            # Gradient descent parameter update
            # epsilon = learning_rate_SGD((i_t*num_examples + j+1)/2000, tau0 = 0.3, kap = 0.55)
            for i in range(len(W_h)):
                W_h[i] += -epsilon * dW_h[i]
                b_h[i] += -epsilon * db_h[i]

            # Assign new parameters to the model
            model = {'W_h': W_h, 'b_h': b_h}

        temp_loss = calculate_loss_batch(model, X, y, reg_lambda)
        if np.isnan(temp_loss):
            print('Loss Nan')
            break
        loss_T.append(calculate_loss_batch(model, X, y, reg_lambda))
        try:
            if np.abs(loss_T[-1] - loss_T[-2]) < converge_error:
                print('Convergence reached!')
                break
        except: pass
        # Optionally print the loss.
        # This is expensive because it uses the whole dataset, so we don't want to do it too often.
        if print_loss:# and i_t % 10 == 0:
            print(
                "After epoch %i - Loss: %f; Error rate: %f" % (i_t, loss_T[-1], calculate_error(model, X, y))
            )

    return (model,loss_T)

rt = build_model_SGD([4,4,4], X[0:600,:], y[0:600], reg_lambda = 0, converge_error = 1e-5, print_loss=True)
model = rt[0]
plt.plot(rt[1])
calculate_error(model, X[600:800,:], y[600:800])

'''Following part is the implementation on 4 datasets in HW2'''

data1_train = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data1_train.csv', sep=' ', header=None)
data2_train = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data2_train.csv', sep=' ', header=None)
data3_train = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data3_train.csv', sep=' ', header=None)
data4_train = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data4_train.csv', sep=' ', header=None)
data1_valid = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data1_validate.csv', sep=' ', header=None)
data2_valid = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data2_validate.csv', sep=' ', header=None)
data3_valid = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data3_validate.csv', sep=' ', header=None)
data4_valid = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data4_validate.csv', sep=' ', header=None)
data1_test = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data1_test.csv', sep=' ', header=None)
data2_test = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data2_test.csv', sep=' ', header=None)
data3_test = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data3_test.csv', sep=' ', header=None)
data4_test = pd.read_csv(r'C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data\data4_test.csv', sep=' ', header=None)

def HW2_dt_convert(df):
    X = np.asarray(df.iloc[:,0:np.size(df,1)-1])
    y = np.asarray(df.iloc[:,-1])
    y = (y * (y>0)).astype(int)
    return (X,y)

d1_train = HW2_dt_convert(data1_train)
d1_test = HW2_dt_convert(data1_train)
rt = build_model_SGD([4,4], d1_train[0], d1_train[1], reg_lambda = 0, converge_error = 1e-5, print_loss=True)
calculate_error(rt[0],d1_test[0],d1_test[1])

d2_train = HW2_dt_convert(data2_train)
d2_test = HW2_dt_convert(data2_train)
rt = build_model_SGD([4,4], d2_train[0], d2_train[1], reg_lambda = 0, converge_error = 1e-5, print_loss=True)
calculate_error(rt[0],d2_test[0],d2_test[1])

d3_train = HW2_dt_convert(data3_train)
d3_test = HW2_dt_convert(data3_train)
rt = build_model_SGD([4,4], d3_train[0], d3_train[1], reg_lambda = 0, converge_error = 1e-5, print_loss=True)
calculate_error(rt[0],d3_test[0],d3_test[1])

d4_train = HW2_dt_convert(data4_train)
d4_test = HW2_dt_convert(data4_train)
rt = build_model_SGD([4,4], d4_train[0], d4_train[1], reg_lambda = 0, converge_error = 1e-5, print_loss=True)
calculate_error(rt[0],d4_test[0],d4_test[1])

'''The following part is the implementation on the MNIST dataset'''
num_samples = 200

MNIST_X = np.zeros((num_samples*10,784))
MNIST_y = np.zeros(num_samples*10)
for i in range(10):
    name = '\mnist_digit_'+str(i)+'.csv'
    temp_dir = r"C:\Users\Jintai\Dropbox (MIT)\_daydayup\6.867_Machine_Learning\hw2\code\data"
    temp = pd.read_csv(temp_dir+name, sep=' ', header=None)
    MNIST_X[i*num_samples:(i+1)*num_samples,:] = np.asarray(temp.iloc[0:num_samples,:])
    MNIST_y[i*num_samples:(i+1)*num_samples] = i
MNIST_X = 2 * MNIST_X / 255 - 1
MNIST_y = MNIST_y.astype(int)

rt = build_model_SGD([5], MNIST_X, MNIST_y, reg_lambda = 0.01, converge_error = 1e-5,
                     num_passes=20 ,epsilon=0.6, print_loss=True)
