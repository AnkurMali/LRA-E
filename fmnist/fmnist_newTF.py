import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow import keras
#from tensorflow.examples.tutorials.mnist import input_data
import time


# random seed to get the consistent result
tf.random.set_seed(1234)

# random seed to get the consistent result
tf.random.set_seed(1234)

#data = input_data.read_data_sets("data/MNIST_data/", one_hot=True)
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

X_val = X_train[58000:60000]
X_train = X_train[0:58000]
y_val = y_train[58000:60000]
y_train = y_train[0:58000]


X_train = X_train.reshape(58000, 28*28)
X_val = X_val.reshape(2000, 28*28)
X_test = X_test.reshape(10000, 28*28)

X_train = X_train/255
X_val = X_val/255
X_test = X_test/255

y_train = tf.keras.utils.to_categorical(y_train, num_classes=10) # Other function is tf.one_hot(y_train,depth=10)
y_val = tf.keras.utils.to_categorical(y_val, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
print(y_train.shape)
minibatch_size = 50
learning_rate = 0.001

## model 1
size_input = 784 # MNIST data input (img shape: 28*28)
size_hidden = 256
size_output = 10 # MNIST total classes (0-9 digits)
beta = 0.1
gamma = 1.0

l_size = [784, 256, 256, 256, 256, 10]
#print(len(l_size))

class MLP(object):
    def __init__(self, layer_sizes, device=None):
        """
        layer_sizes: list of int, sizes of each layer including input and output layers
        device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
        """
        self.layer_sizes = layer_sizes
        self.device = device

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        self.E = []

        for i in range(len(layer_sizes) - 1):
            weight = tf.Variable(tf.random.normal([layer_sizes[i], layer_sizes[i + 1]], stddev=0.1, dtype=tf.float32), name=f"W{i + 1}")
            bias = tf.Variable(tf.zeros([layer_sizes[i + 1]], dtype=tf.float32), name=f"b{i + 1}")
            if i>0:
              error_weight = tf.Variable(tf.random.normal([layer_sizes[i+1], layer_sizes[i]], stddev=1.0, dtype=tf.float32), name=f"E{i + 1}")
              self.E.append(error_weight)
            self.weights.append(weight)
            self.biases.append(bias)
        # Define variables to be updated during backpropagation
        self.variables_w = self.weights + self.E

    def compute_output(self, X):
        """
        Custom method to obtain output tensor during forward pass
        """
        X_tf = tf.cast(X, dtype=tf.float32)
        activations = X_tf
        for i in range(len(self.weights) - 1):
            #pre_activation = tf.matmul(activations, self.weights[i]) + self.biases[i]
            pre_activation = tf.matmul(activations, self.weights[i]) 
            activations = tf.nn.tanh(pre_activation)
        
        #logits = tf.matmul(activations, self.weights[-1]) + self.biases[-1]
        logits = tf.matmul(activations, self.weights[-1])
        self.z_last = tf.nn.softmax(logits)
        return logits

    def compute_lra_updates(self, X_train, Y_train, learning_rate=2e-4, beta=0.05, gamma=1.0):
        """
        LRA_update
        """
        optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate)
        # Forward pass
        logits = self.compute_output(X_train)
        Y_train = tf.cast(Y_train, dtype=tf.float32)
        e_last = tf.subtract(self.z_last, Y_train)
        #print(e_last.shape)
        
        # Initialize variables
        e = [None] * (len(self.layer_sizes) - 1)
        d = [None] * (len(self.layer_sizes) - 1)
        y_z = [None] * (len(self.layer_sizes) - 1)
        dW = [None] * (len(self.layer_sizes) - 1)
        dW_e = [None] * (len(self.layer_sizes) - 1)

        e[-1] = e_last
        batch_size = tf.shape(X_train)[0]
        for i in range(len(self.layer_sizes) - 2, 0, -1):
            d[i] = tf.matmul(e[i], self.E[i-1])
            d_b = tf.multiply(d[i], beta)
            y_z[i] = tf.nn.tanh(tf.subtract(self.compute_pre_activation(i-1, X_train), (d_b)))
            e[i-1] = tf.subtract(self.compute_activation(i - 1, X_train), y_z[i])
        
        for i in range(len(self.layer_sizes) - 1):
            if i == 0:
                dW[i] = tf.matmul(tf.cast(tf.transpose(e[i]), tf.float32), tf.cast(X_train, tf.float32)) / tf.cast(batch_size, tf.float32)
                dW[i] = tf.transpose(dW[i])
            else:
                dW[i] = tf.matmul(tf.transpose(e[i]), self.compute_activation(i - 1, X_train)) / tf.cast(batch_size, tf.float32)
                dW[i] = tf.transpose(dW[i])
            dW_e[i] = tf.multiply(tf.transpose(dW[i]), gamma)

        grads_w = [dW[i] for i in range(len(self.layer_sizes) - 1)] + dW_e[1:]
        optimizer.apply_gradients(zip(grads_w, self.variables_w))

    def compute_pre_activation(self, layer_index, X):
        """
        Compute pre-activation output of a specific layer during forward pass
        """
        activations = tf.cast(X, dtype=tf.float32)
        for i in range(layer_index):
            #pre_activation = tf.matmul(activations, tf.cast(self.weights[i], dtype=tf.float32)) + tf.cast(self.biases[i], dtype=tf.float32)
            pre_activation = tf.matmul(activations, tf.cast(self.weights[i], dtype=tf.float32))
            activations = tf.nn.tanh(pre_activation)
        return tf.matmul(activations, tf.cast(self.weights[layer_index], dtype=tf.float32))
        #return tf.matmul(activations, tf.cast(self.weights[layer_index], dtype=tf.float32)) + tf.cast(self.biases[layer_index], dtype=tf.float32)

    def compute_activation(self, layer_index, X):
        """
        Compute activation output of a specific layer during forward pass
        """
        activations = tf.cast(X, dtype=tf.float32)
        for i in range(layer_index + 1):
            #pre_activation = tf.matmul(activations, tf.cast(self.weights[i], dtype=tf.float32)) + tf.cast(self.biases[i], dtype=tf.float32)
            pre_activation = tf.matmul(activations, tf.cast(self.weights[i], dtype=tf.float32))
            activations = tf.nn.tanh(pre_activation)
        return activations

    def forward(self, X):
        """
        forward pass
        X: Tensor, inputs
        """
        if self.device is not None:
            try:
                with tf.device('gpu:0' if self.device == 'gpu' else 'cpu'):
                    self.y = self.compute_output(X)
            except:
                self.y = self.compute_output(X)  # Sometimes windows hardware or ubuntu 14.04 throw error with python2.7
        else:
            self.y = self.compute_output(X)
        # self.y = self.compute_output(X)   
        return self.y

    def loss(self, y_pred, y_true):
        '''
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        '''
        y_true_tf = tf.cast(y_true, dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_pred_tf, labels=y_true_tf))

def accuracy_function(yhat,true_y):
  yhat = tf.nn.softmax(yhat)
  correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(true_y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy

# Initialize model using GPU
mlp_on_cpu = MLP(l_size, device='gpu')

num_epochs = 100

time_start = time.time()
num_train = 58000

"""
for epoch in range(num_epochs):
        train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(25, seed=epoch*(1234))\
           .shuffle(buffer_size=1000)\
           .batch(batch_size=minibatch_size)
        loss_total = tf.Variable(0, dtype=tf.float32)
        epoch_k = 0.0
        for inputs, outputs in train_ds:
            #print(inputs.shape)
            #break
            #print(outputs.shape)
            preds = mlp_on_cpu.forward(inputs)
            loss_total = loss_total + mlp_on_cpu.loss(preds, outputs)
            #print(loss_total)
            epoch_k+=1
            mlp_on_cpu.compute_lra_updates(inputs, outputs)
        print('Number of Epoch = {} - loss:= {:.4f}'.format(epoch + 1, loss_total.numpy() / epoch_k))
        preds = mlp_on_cpu.compute_output(X_train)
        accuracy_train = accuracy_function(preds,y_train)
        accuracy_train = accuracy_train * 100
        print ("Training Accuracy = {}".format(accuracy_train.numpy()))
        
        preds_val = mlp_on_cpu.compute_output(X_val)
        accuracy_val = accuracy_function(preds_val,y_val)
        accuracy_val = accuracy_val * 100
        print ("Validation Accuracy = {}".format(accuracy_val.numpy()))
"""
best_accuracy = 0.0
best_weights = None

for epoch in range(num_epochs):
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(25, seed=epoch*(1234))\
       .shuffle(buffer_size=1000)\
       .batch(batch_size=minibatch_size)
    loss_total = tf.Variable(0, dtype=tf.float32)
    epoch_k = 0.0
    for inputs, outputs in train_ds:
        preds = mlp_on_cpu.forward(inputs)
        loss_total = loss_total + mlp_on_cpu.loss(preds, outputs)
        epoch_k += 1
        mlp_on_cpu.compute_lra_updates(inputs, outputs)

    # Compute and print training accuracy
    preds = mlp_on_cpu.compute_output(X_train)
    accuracy_train = accuracy_function(preds, y_train) * 100
    print("Epoch {}: Training Accuracy = {:.2f}%".format(epoch + 1, accuracy_train.numpy()))

    # Compute and print validation accuracy
    preds_val = mlp_on_cpu.compute_output(X_val)
    accuracy_val = accuracy_function(preds_val, y_val) * 100
    print("Epoch {}: Validation Accuracy = {:.2f}%".format(epoch + 1, accuracy_val.numpy()))

    # Check if the validation accuracy is the best we've seen so far
    if accuracy_val > best_accuracy:
        best_accuracy = accuracy_val
        #best_weights = mlp_on_cpu.get_weights()  # Assuming get_weights() retrieves model weights

    # Print the current and best validation accuracy
    print('Current Validation Accuracy: {:.2f}% - Best Validation Accuracy: {:.2f}%'.format(accuracy_val.numpy(), best_accuracy.numpy()))
    
# test accuracy
preds_test = mlp_on_cpu.compute_output(X_test)
accuracy_test = accuracy_function(preds_test,y_test)
# To keep sizes compatible with model
accuracy_test = accuracy_test * 100
print ("Test Accuracy = {}".format(accuracy_test.numpy()))

        
time_taken = time.time() - time_start
print('\nTotal time taken (in seconds): {:.2f}'.format(time_taken))
#For per epoch_time = Total_Time / Number_of_epochs
