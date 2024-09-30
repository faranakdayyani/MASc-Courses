# -*- coding: utf-8 -*-
"""
@author: Faranak Dayyani
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Load the data
def load_data():
    with np.load("notMNIST.npz") as data:
        data, targets = data["images"], data["labels"]
        
        np.random.seed(521)
        rand_idx = np.arange(len(data))
        np.random.shuffle(rand_idx)
        
        data = data[rand_idx] / 255.0
        targets = targets[rand_idx].astype(int)
        
        train_data, train_target = data[:10000], targets[:10000]
        valid_data, valid_target = data[10000:16000], targets[10000:16000]
        test_data, test_target = data[16000:], targets[16000:]
    return train_data, valid_data, test_data, train_target, valid_target, test_target


def convert_onehot(train_target, valid_target, test_target):
    new_train = np.zeros((train_target.shape[0], 10))
    new_valid = np.zeros((valid_target.shape[0], 10))
    new_test = np.zeros((test_target.shape[0], 10))

    for item in range(0, train_target.shape[0]):
        new_train[item][train_target[item]] = 1
    for item in range(0, valid_target.shape[0]):
        new_valid[item][valid_target[item]] = 1
    for item in range(0, test_target.shape[0]):
        new_test[item][test_target[item]] = 1
    return new_train, new_valid, new_test



def shuffle(data, target):
    np.random.seed(421)
    rand_idx = np.random.permutation(len(data))
    return data[rand_idx], target[rand_idx]


# Implementation of a neural network using only Numpy - trained using gradient descent with momentum
    

def relu(x): 
    re_lu = np.maximum(x,0)
    return re_lu


def softmax(x): 

    x = x - x.max(axis=1, keepdims=True) 
    z = np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)

    return z
    

def compute_layer(x, w, b):

    product = x @ w + b

    return product

def average_ce(target, prediction): 

    avg_ce = -np.mean(target*np.log(prediction))

    return avg_ce


def grad_ce(target, logits):

    gr_ce = (softmax(logits) - target)/target.shape[0]
    
    return gr_ce

def grad_wrt_wO(h, p, y):

    p_y = grad_ce(y, p)
    h_tran = np.transpose(h)
    gradwO = np.matmul(h_tran, p_y)

    return gradwO


def grad_wrt_bO(p, y):

    p_y = grad_ce(y, p)
    ones_shape = np.ones((1,y.shape[0]))
    gradbO = np.matmul(ones_shape, p_y)

    return gradbO


def grad_wrt_wh(p, y, x, h_wh, w_O):
    
    h_wh [h_wh > 0] = 1
    h_wh [h_wh < 0] = 0
    
    x_tran = x.transpose()
    p_y = grad_ce(y, p)

    part1 = np.matmul(p_y, np.transpose(w_O))
    part2 = h_wh * part1
    
    grad_wh = np.matmul(x_tran, part2)
    
    return grad_wh



def grad_wrt_bh(p, y, x, h_wh, w_O):
    
    h_wh [h_wh > 0] = 1
    h_wh [h_wh < 0] = 0
    
    p_y = grad_ce(y, p)    
    ones_shape = np.ones((x.shape[0], 1))
    ones_shape_tran = ones_shape.transpose()
    
    part1 = np.matmul(p_y, np.transpose(w_O))
    part2 = h_wh * part1
    
    grad_bh = np.matmul(ones_shape_tran, part2)
     
    return grad_bh



def learning(train_data, train_target, valid_data, valid_target, w_o, b_o, v_o, w_h, b_h, v_h, epochs, gamma, alpha):
    
    global i, b_o_1, b_h_1

    # creating array variables for training and validation loss and accuracies
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    
    b_o_1 = b_o
    b_h_1 = b_h
    
    for i in range(epochs):
        
        # ----------- forward propagation --------------------
        # training data: calculating loss and accuracy
        #               hidden layer
        producttrain_h = compute_layer(train_data, w_h, b_h)
        re_lu_train_h = relu(producttrain_h)
        #               output layer
        producttrain_o = compute_layer(re_lu_train_h, w_o, b_o)
        y_hat_train = softmax(producttrain_o)
        avg_ce_train = average_ce(new_train, y_hat_train)
        train_loss.append(avg_ce_train)
        acc1 = np.argmax(y_hat_train, axis=1)
        acc11 = np.argmax(new_train, axis=1)
        acc111 = np.sum(acc1 == acc11)/train_data.shape[0]
        acc111 = acc111*100
        train_acc.append(acc111)
        
        # validation data: calculating loss and accuracy
        #               hidden layer
        productvalid_h = compute_layer(valid_data, w_h, b_h)
        re_lu_valid_h = relu(productvalid_h)
        #               output layer
        productvalid_o = compute_layer(re_lu_valid_h, w_o, b_o)
        y_hat_valid = softmax(productvalid_o)
        avg_ce_valid = average_ce(new_valid, y_hat_valid)
        valid_loss.append(avg_ce_valid)
        acc2 = np.argmax(y_hat_valid, axis=1)
        acc22 = np.argmax(new_valid, axis=1)
        acc222 = np.sum(acc2 == acc22)/valid_data.shape[0]
        acc222 = acc222*100
        valid_acc.append(acc222)
        
        
        # ----------- Backpropagation --------------------
        #               Output layer
        gradwO = grad_wrt_wO(re_lu_train_h, producttrain_o, new_train)
        v_o = (gamma * v_o) + (alpha * gradwO)
        w_o = w_o - v_o
        
        gradbO = grad_wrt_bO(producttrain_o, new_train)
        b_o_1 = (gamma * b_o_1) + (alpha * gradbO)
        b_o = b_o - b_o_1
        
        #               Hidden layer
        grad_wh = grad_wrt_wh(producttrain_o, new_train, train_data, producttrain_h, w_o)
        v_h = (gamma * v_h) + (alpha * grad_wh)
        v_h = w_h - v_h
        
        grad_bh = grad_wrt_bh(producttrain_o, new_train, train_data, producttrain_h, w_o)
        b_h_1 = (gamma * b_h_1) + (alpha * grad_bh)
        b_h = b_h - b_h_1
        
        print("Going through the loop: Epoch ", i+1,"/", epochs)        
        
    return w_o, b_o, w_h, b_h, train_loss, train_acc, valid_loss, valid_acc


if __name__ == "__main__":
    train_data, valid_data, test_data, train_target, valid_target, test_target = load_data()
    train_data = train_data.reshape(10000, 784)
    valid_data = valid_data.reshape(6000, 784)
    test_data = test_data.reshape(2724, 784)

    new_train, new_valid, new_test = convert_onehot(train_target, valid_target, test_target)
        
    # --------------------------------------- section 1.3 ------------------------------------------------------
    epochs = 200
    H = 1000
    gamma = 0.99
    alpha = 0.1
    
    # initializing weights with Xaiver initialization scheme:
    # bias set to 0
    # initial momentum value of 1e-5
    
    # -------- output layer initialization -------------
    limit_o = np.sqrt(2/(H+10))
    w_o = np.random.normal(0.0, limit_o, (H,10))
    v_o = np.full((H,10), 1e-5)
    b_o = np.zeros((1,10))
    
    # -------- hidden layer initialization -------------
    limit_h = np.sqrt(2/(train_data.shape[1]+H))
    w_h = np.random.normal(0.0, limit_h, (train_data.shape[1],H))
    v_h = np.full((train_data.shape[1], H), 1e-5)
    b_h = np.zeros((1, H))
    
    w_o, b_o, w_h, b_h, train_loss, train_acc, valid_loss, valid_acc = learning(train_data, new_train, valid_data, new_valid, w_o, b_o,
                                                                                v_o, w_h, b_h, v_h, epochs, gamma, alpha/10)    
    
    
    # Training data result:
    # ------ hidden layer ------
    product_h_train = compute_layer(train_data, w_h, b_h)
    relu_h_train = relu(product_h_train)
    # ------ output layer ------
    product_o_train = compute_layer(relu_h_train, w_o, b_o)
    yhat_train = softmax(product_o_train)
    avgce_train = average_ce(new_train, yhat_train)
    train_loss.append(avgce_train)
    acctrain1 = np.argmax(yhat_train, axis=1)
    acctrain11 = np.argmax(new_train, axis=1)
    acctrain111 = np.sum(acctrain1 == acctrain11)/train_data.shape[0]
    acctrain111 = acctrain111*100
    train_acc.append(acctrain111)
    print(" -------- Training Data ----------")
    print("Training loss is:", train_loss[-1])
    print("Training accuracy is:", train_acc[-1]," %")
    
    # Validation data result:
    # ------ hidden layer ------
    product_h_valid = compute_layer(valid_data, w_h, b_h)
    relu_h_valid = relu(product_h_valid)
    # ------ output layer ------
    product_o_valid = compute_layer(relu_h_valid, w_o, b_o)
    yhat_valid = softmax(product_o_valid)
    avgce_valid = average_ce(new_valid, yhat_valid)
    valid_loss.append(avgce_valid)
    accvalid1 = np.argmax(yhat_valid, axis=1)
    accvalid11 = np.argmax(new_valid, axis=1)
    accvalid111 = np.sum(accvalid1 == accvalid11)/valid_data.shape[0]
    accvalid111 = accvalid111*100
    valid_acc.append(accvalid111)
    print("\n -------- Validation Data ----------")
    print("Validation loss is:", valid_loss[-1])
    print("Validation accuracy is:", valid_acc[-1]," %")
    
    
    
    # Testing data result:
    test_loss = []
    test_acc = []
    # ------ hidden layer ------
    product_h_test = compute_layer(test_data, w_h, b_h)
    relu_h_test = relu(product_h_test)
    # ------ output layer ------
    product_o_test = compute_layer(relu_h_test, w_o, b_o)
    yhat_test = softmax(product_o_test)
    avgce_test = average_ce(new_test, yhat_test)
    test_loss.append(avgce_test)
    acctest1 = np.argmax(yhat_test, axis=1)
    acctest11 = np.argmax(new_test, axis=1)
    acctest111 = np.sum(acctest1 == acctest11)/test_data.shape[0]
    acctest111 = acctest111*100
    test_acc.append(acctest111)
    print("\n -------- Testing Data ----------")
    print("Testing loss is:", test_loss[-1])
    print("Testing accuracy is:", test_acc[-1]," %")
    

    
    # plotting loss for training and validation data:
    xaxis = np.linspace(0, epochs, num=epochs)
    plt.plot(xaxis, train_loss[0:epochs])
    plt.plot(xaxis, valid_loss[0:epochs])
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Number of Epochs')
    plt.legend(['train_data', 'valid_data'])
    plt.grid()
    plt.show()
    plt.figure()
        
    # plotting accuracy for training and validation data:
    plt.plot(xaxis, train_acc[0:epochs])
    plt.plot(xaxis, valid_acc[0:epochs])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Number of Epochs')
    plt.legend(['train_data','valid_data'])
    plt.grid()
    plt.show()
    