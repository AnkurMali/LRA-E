# -*- coding: utf-8 -*-
"""
Author:-aam35
Procedure to train 4 layer MLP using LRA-E on fashion-mnist 
"""

import numpy as np
import os
import sys
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import time
tf.enable_eager_execution()
tf.executing_eagerly()

# random seed to get the consistent result
tf.random.set_random_seed(1234)

data = input_data.read_data_sets("data/FMNIST_data/", one_hot=True)


minibatch_size = 50
learning_rate = 0.001

## model 1
size_input = 784 # FMNIST data input (img shape: 28*28)
size_hidden = 256
size_output = 10 # FMNIST total classes (0-9 digits)
beta = 0.1
gamma = 1.0


# Define class to build mlp model
class MLP(object):
    def __init__(self, size_input, size_hidden, size_output, device=None):
        """
        size_input: int, size of input layer
        size_hidden: int, size of hidden layer
        size_output: int, size of output layer
        device: str or None, either 'cpu' or 'gpu' or None. If None, the device to be used will be decided automatically during Eager Execution
        """
        self.size_input, self.size_hidden, self.size_output, self.device =\
        size_input, size_hidden, size_output, device
    
        # Initialize weights between input layer and hidden layer
        self.W1 = tf.Variable(tf.random.normal([self.size_input, self.size_hidden],stddev=0.1),name="W1")
        # Initialize biases for hidden layer
        self.b1 = tf.Variable(tf.zeros([1, self.size_hidden]), name = "b1")
        # Initialize weights between hidden layer and output layer
        self.W2 = tf.Variable(tf.random.normal([self.size_hidden, self.size_hidden],stddev=0.1),name="W2")
        # Initialize biases for output layer
        self.b2 = tf.Variable(tf.random.normal([1, self.size_hidden]),name="b2")
        
        self.W3 = tf.Variable(tf.random.normal([self.size_hidden, self.size_hidden],stddev=0.1),name="W3")
        # Initialize biases for output layer
        self.b3 = tf.Variable(tf.random.normal([1, self.size_hidden]),name="b3")
        
        self.W4 = tf.Variable(tf.random.normal([self.size_hidden, self.size_output],stddev=0.1),name="W4")
        # Initialize biases for output layer
        self.b4 = tf.Variable(tf.random.normal([1, self.size_output]),name="b4")
        
        self.E2 = tf.Variable(tf.random.normal([self.size_hidden, self.size_hidden],stddev=1.0),name="E2")
        self.E3 = tf.Variable(tf.random.normal([self.size_hidden, self.size_hidden],stddev=1.0),name="E3")
        self.E4 = tf.Variable(tf.random.normal([self.size_output, self.size_hidden],stddev=1.0),name="E4")
    

        
        # Define variables to be updated during backpropagation
        self.variables_w = [self.W1, self.W2,self.W3,self.W4,self.E2,self.E3,self.E4]
        #self.variables_e = [self.E2,self.E3,self.E4]
        
    
    # prediction
    def forward(self, X):
        """
        forward pass
        X: Tensor, inputs
        """
        if self.device is not None:
	    try:	
                with tf.device('gpu:0' if self.device=='gpu' else 'cpu'):
                    self.y = self.compute_output(X)
            except:
                 self.y = self.compute_output(X) #Sometimes windows hardware or ubuntu 14.04 throw error with python2.7
        else:
            self.y = self.compute_output(X)
        #self.y = self.compute_output(X)   
        return self.y
    
    ## loss function
    def loss(self, y_pred, y_true):
        '''
        y_pred - Tensor of shape (batch_size, size_output)
        y_true - Tensor of shape (batch_size, size_output)
        '''
        y_true_tf = tf.cast(tf.reshape(y_true, (-1, self.size_output)), dtype=tf.float32)
        y_pred_tf = tf.cast(y_pred, dtype=tf.float32)
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y_pred_tf, labels=y_true_tf))
        
    ##BP backward pass
    def backward(self, X_train, y_train):
        """
        backward pass
        """
        # optimizer
        # Test with SGD,Adam, RMSProp
        optimizer =  tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        #optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        with tf.GradientTape() as tape:
            predicted = self.forward(X_train)
            current_loss = self.loss(predicted, y_train)
            #print(current_loss)
        #print(current_loss.shape)
        grads = tape.gradient(current_loss, self.variables_w)
        optimizer.apply_gradients(zip(grads, self.variables_w),
                              global_step=tf.compat.v1.train.get_or_create_global_step())
        
    ## forward pass to get pre and post activations    
    def compute_output(self, X):
        """
        Custom method to obtain output tensor during forward pass
        """
        # Cast X to float32
        X_tf = tf.cast(X, dtype=tf.float32)
        #Remember to normalize your dataset before moving forward
        # Compute values in hidden layer
        self.what = tf.matmul(X_tf, self.W1) + self.b1
        self.hhat = tf.nn.tanh(self.what)
        self.what1 = tf.matmul(self.hhat,self.W2)+ self.b2
        self.hhat1 = tf.nn.tanh(self.what1)
        self.what2 = tf.matmul(self.hhat1,self.W3) + self.b3
        self.hhat2 = tf.nn.tanh(self.what2)
        # Compute output
        self.logits = tf.matmul(self.hhat2, self.W4) + self.b4
        self.z4 = tf.nn.softmax(self.logits)
        #Now consider two things , First look at inbuild loss functions if they work with softmax or not and then change this
        #Second add tf.Softmax(output) and then return this variable
        #print(output)
        return (self.logits)
        #return output
        
    def compute_lra_updates(self, X_train, Y_train):
        """
        LRA_update
        
        """
        #Compute targets/updates
        
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate)
        
        e4 = tf.subtract(self.z4,(Y_train))
        d3 = tf.matmul(tf.transpose(self.E4),tf.transpose(e4))
        d3_b = tf.multiply(d3,beta) 
        #print("second_subtract")
        y3_z = tf.nn.tanh(tf.subtract(tf.transpose(self.what2),(d3_b)))
        
        e3 = tf.subtract(self.hhat2,tf.transpose(y3_z))
        d2 = tf.matmul(self.E3,tf.transpose(e3))
        d2_b = tf.multiply(d2,beta)
        y2_z = tf.nn.tanh(tf.subtract(self.what1,tf.transpose(d2_b)))
        
        e2 = tf.subtract(self.hhat1,y2_z)
        d1 = tf.matmul(self.E2,tf.transpose(e2))
        d1_b = tf.multiply(d1,beta)
        y1_z = tf.nn.tanh(tf.subtract(self.what,tf.transpose(d1_b)))
        
        e1 = tf.subtract(self.hhat,y1_z)
        
        
        dW4 = tf.matmul(e4,self.hhat2,transpose_a=True)
        dW3 = tf.matmul(e3,self.hhat1,transpose_a = True)
        dW2 = tf.matmul(e2,self.hhat,transpose_a = True)
        dW1 = tf.matmul(e1,X_train,transpose_a = True)
        
        #dW4 = dW4/(tf.norm(dW4) + 0.00000001)
        #dW3 = dW3/(tf.norm(dW3) + 0.00000001)
        #dW2 = dW2/(tf.norm(dW2) + 0.00000001)
        #dW1 = dW1/(tf.norm(dW1) + 0.00000001)
        #print(dW4.shape)
        #print(dW3.shape)
        #print(dW2.shape)
        #print(dW1.shape)
        
        
        
        dW4_e = (dW4)
        dW4_e = tf.multiply(dW4_e,gamma)
        
        dW3_e = (dW3)
        dW3_e = tf.multiply(dW3_e,gamma)
        
        dW2_e = (dW2)
        dW2_e = tf.multiply(dW2_e,gamma)
        
        grads_w = [tf.transpose(dW1),tf.transpose(dW2),tf.transpose(dW3),tf.transpose(dW4),dW2_e, dW3_e, dW4_e]
        
#         dW_4 = tf.multiply(tf.transpose(dW4),0.001)
#         dW_3 = tf.multiply(tf.transpose(dW3),0.001)
#         dW_2 = tf.multiply(tf.transpose(dW2),0.001)
#         dW_1 = tf.multiply(tf.transpose(dW1),0.001)
        
        
        optimizer.apply_gradients(zip(grads_w, self.variables_w),global_step=tf.compat.v1.train.get_or_create_global_step())
        

def accuracy_function(yhat,true_y):
  correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(true_y, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy

# Initialize model using GPU
mlp_on_cpu = MLP(size_input, size_hidden, size_output, device='gpu')

num_epochs = 100

time_start = time.time()
num_train = 55000


for epoch in range(num_epochs):
        train_ds = tf.data.Dataset.from_tensor_slices((data.train.images, data.train.labels)).map(lambda x, y: (x, tf.cast(y, tf.float32)))\
           .shuffle(buffer_size=1000)\
           .batch(batch_size=minibatch_size)
        loss_total = tf.Variable(0, dtype=tf.float32)
        for inputs, outputs in train_ds:
            preds = mlp_on_cpu.forward(inputs)
            loss_total = loss_total + mlp_on_cpu.loss(preds, outputs)
            mlp_on_cpu.compute_lra_updates(inputs, outputs)
        print('Number of Epoch = {} - loss:= {:.4f}'.format(epoch + 1, loss_total.numpy() / num_train))
        preds = mlp_on_cpu.compute_output(data.train.images)
        accuracy_train = accuracy_function(preds,data.train.labels)
        accuracy_train = accuracy_train * 100
        print ("Training Accuracy = {}".format(accuracy_train.numpy()))
        
        preds_val = mlp_on_cpu.compute_output(data.validation.images)
        accuracy_val = accuracy_function(preds_val,data.validation.labels)
        accuracy_val = accuracy_val * 100
        print ("Validation Accuracy = {}".format(accuracy_val.numpy()))
 
    
# test accuracy
preds_test = mlp_on_cpu.compute_output(data.test.images)
accuracy_test = accuracy_function(preds_test,data.test.labels)
# To keep sizes compatible with model
accuracy_test = accuracy_test * 100
print ("Test Accuracy = {}".format(accuracy_test.numpy()))

        
time_taken = time.time() - time_start
print('\nTotal time taken (in seconds): {:.2f}'.format(time_taken))
#For per epoch_time = Total_Time / Number_of_epochs
