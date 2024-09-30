# -*- coding: utf-8 -*-
"""
@author: Faranak Dayyani
"""



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def loadData():
    with np.load('notMNIST.npz') as dataset:
        Data, Target = dataset['images'], dataset['labels']
        posClass = 2
        negClass = 9
        dataIndx = (Target==posClass) + (Target==negClass)
        Data = Data[dataIndx]/255.
        Target = Target[dataIndx].reshape(-1, 1)
        Target[Target==posClass] = 1
        Target[Target==negClass] = 0
        np.random.seed(421)
        randIndx = np.arange(len(Data))
        np.random.shuffle(randIndx)
        Data, Target = Data[randIndx], Target[randIndx]
        trainData, trainTarget = Data[:3500], Target[:3500]
        validData, validTarget = Data[3500:3600], Target[3500:3600]
        testData, testTarget = Data[3600:], Target[3600:]
    return trainData, validData, testData, trainTarget, validTarget, testTarget


# sigmoid function     
def sigmoid(x,w):
    z = np.dot(x, w)
    s = 1/(1 + np.exp(-z))
    
    return s


def loss(w, b, x, y, reg):
    
    global LW, L, LCE,LCE_mean, y_hat
    
    y_hat = sigmoid(x, w)
    
    w = np.delete(w,-1)
    w = w.reshape(1,784)
    LCE = ((-1*y)*(np.log(y_hat))) - ((1 - y)*np.log(1- y_hat))
    LCE_mean = np.mean(LCE) #cross entropy loss value
    
    w_norm = np.linalg.norm(w)  #takes the norm L2
    LW = (reg/2)*((w_norm)**2) #regularization loss
    L = LCE_mean + LW   #total loss
    
    return L
    

def grad_loss(w, b, x, y, reg):
    
    global Lgrad_wrt_w, Lgrad_wrt_b
    
    y_hat = sigmoid(x, w)
    
    m = len(x)
    y_result = y_hat - y
    x_tran = x.T
    Lgrad_wrt_w = ((1/m)*(np.dot(x_tran, y_result))) + (reg*w)
    Lgrad_wrt_b = Lgrad_wrt_w[784,0]    
    Lgrad_wrt_w = np.delete(Lgrad_wrt_w,-1)  
    Lgrad_wrt_w = Lgrad_wrt_w.reshape(784,1)
    return Lgrad_wrt_w, Lgrad_wrt_b
    


def grad_descent(w, b, x, y, xvalid, yvalid, xtest, ytest, alpha, epochs, reg, error_tol):
    
    global w_updated, b_updated, Ltrain, Lvalid, Larr_train, xaxis, Larr_train_a, Larr_valid, Larr_valid_a, y_hat_train1, y_hat_train, y_hat_valid
    Larr_train = 0
    Larr_valid = 0
    Aarr_train = 0
    Aarr_valid = 0
    
    for i in range (epochs):
        
        # loss data
        Ltrain = loss(w, b, x, y, reg)
        Lvalid = loss(w, b, xvalid, yvalid, reg)   
        Ltest = loss(w, b, xtest, ytest, reg)
        
        # accuracy data
        y_hat_train1 = sigmoid(x, w)
        y_hat_train = sigmoid(x, w)
        y_hat_train[y_hat_train >= 0.5] = 1
        y_hat_train[y_hat_train < 0.5] = 0
        Atrain = np.mean(y_hat_train == y)
        Atrain = Atrain*100
        
        y_hat_valid = sigmoid(xvalid, w)
        y_hat_valid[y_hat_valid >= 0.5] = 1
        y_hat_valid[y_hat_valid < 0.5] = 0
        Avalid = np.mean(y_hat_valid == yvalid)
        Avalid = Avalid*100
        
        y_hat_test = sigmoid(xtest, w)
        y_hat_test[y_hat_test >= 0.5] = 1
        y_hat_test[y_hat_test < 0.5] = 0
        Atest = np.mean(y_hat_test == ytest)
        Atest = Atest*100
        
        Lgrad_wrt_w, Lgrad_wrt_b = grad_loss(w, b, x, y, reg)
        
    
        b = w[784,0]    
        w = np.delete(w,-1) 
        w = w.reshape(784,1)
        
        w_updated = w - (alpha*Lgrad_wrt_w)
        b_updated = b - (alpha*Lgrad_wrt_b)
        
        w_difference = w_updated - w
        error = np.linalg.norm(w_difference)
              
        if error < error_tol:
            
            print("error is less than error tolerance")
            
            # ---- plotting Loss -----
            # array for plotting train data
            Larr_train = np.append(Larr_train,Ltrain)
            Larr_train_a = Larr_train[1:epochs+1]
        
            #array for plotting validation data
            Larr_valid = np.append(Larr_valid,Lvalid)
            Larr_valid_a = Larr_valid[1:epochs+1]
                        
            xaxis = np.linspace(0, epochs, num=epochs)    
            plt.plot(xaxis,Larr_train_a)
            plt.plot(xaxis,Larr_valid_a)
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Loss vs. Number of Epochs - regularization = 0.5')
            plt.legend(['trainData','validData'])
            plt.grid()
            plt.show()
            plt.figure()
            
            # ---- plotting Accuracy -----
            #array for plotting train data
            Aarr_train = np.append(Aarr_train, Atrain)
            Aarr_train_a = Aarr_train[1:epochs+1]
        
            #array for plotting validation data
            Aarr_valid = np.append(Aarr_valid,Avalid)
            Aarr_valid_a = Aarr_valid[1:epochs+1]
            
            xaxis = np.linspace(0, epochs, num=epochs)    
            plt.plot(xaxis,Aarr_train_a)
            plt.plot(xaxis,Aarr_valid_a)
            plt.xlabel('Epochs')
            plt.ylabel('Accuracy (%)')
            plt.title('Accuracy vs. Number of Epochs - regularization = 0.5')
            plt.legend(['trainData','validData'])
            plt.grid()
            plt.show()
             
            print("error is less than error_tol")
            break;
          
        w = w_updated
        b = b_updated
        w = np.append(w,b)
        w = w.reshape(785,1)
        
        # ---- Loss arrays -----
        # array for plotting train data
        Larr_train = np.append(Larr_train,Ltrain)
        Larr_train_a = Larr_train[1:epochs+1]
        
        #array for plotting validation data
        Larr_valid = np.append(Larr_valid,Lvalid)
        Larr_valid_a = Larr_valid[1:epochs+1]
        
        # ---- Accuracy arrays -----
        #array for plotting train data
        Aarr_train = np.append(Aarr_train, Atrain)
        Aarr_train_a = Aarr_train[1:epochs+1]
            
        #array for plotting validation data
        Aarr_valid = np.append(Aarr_valid,Avalid)
        Aarr_valid_a = Aarr_valid[1:epochs+1]
        
    
    # ---- plotting Loss -----
    xaxis = np.linspace(0, epochs, num=epochs)    
    plt.plot(xaxis,Larr_train_a)
    plt.plot(xaxis,Larr_valid_a)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Number of Epochs - regularization = 0.5')
    plt.legend(['trainData','validData'])
    plt.grid()
    plt.show()
    
    plt.figure()
    
    # ---- plotting Accuracy -----
    xaxis = np.linspace(0, epochs, num=epochs)    
    plt.plot(xaxis,Aarr_train_a)
    plt.plot(xaxis,Aarr_valid_a)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy vs. Number of Epochs - regularization = 0.5')
    plt.legend(['trainData','validData'])
    plt.grid()
    plt.show()
    
    print("Training Loss:",Ltrain)
    print("Validation Loss:",Lvalid)
    print("Testing Loss:",Ltest)
    print("Training Accuracy:",Atrain)
    print("Validation Accuracy:",Avalid)
    print("Testing Accuracy:",Atest)
    
    
    return w_updated, b_updated



if __name__ == "__main__":
    trainData, validData, testData, trainTarget, validTarget, testTarget = loadData()
    
    w = np.zeros((1,785))    #this gives out 1x785
    w_transpose = w.transpose()
    b = w[0,784]
    
    # Training data
    arraytrain = np.ones((3500,1))
    xtrain = trainData.reshape(3500,28*28)   # true data, this gives out 3500x784
    xtrain = np.append(xtrain,arraytrain, axis = 1)
    ytrain = trainTarget
    
    # validation data
    arrayvalid = np.ones((100,1))
    xvalid= validData.reshape(100,28*28)   # true data, this gives out 100x784
    xvalid = np.append(xvalid,arrayvalid, axis = 1)
    yvalid = validTarget
    
    # testing data
    arraytest = np.ones((145,1))
    xtest= testData.reshape(145,28*28)   # true data, this gives out 145x784
    xtest = np.append(xtest,arraytest, axis = 1)
    ytest = testTarget
    
    
    error_tol = 10**-7
    w_updated, b_updated = grad_descent(w_transpose, b, xtrain, ytrain, xvalid, yvalid, xtest, ytest, 0.005, 5000, 0.1, error_tol)
    
   
    
    
    
    
    
    
    
    
    
    
    
    