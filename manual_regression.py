# implement a NN from scratch using just numpy

import numpy as np

X = np.array([1,2,3,4] , dtype=np.float32)
Y = np.array([2,4,6,8] , dtype=np.float32)

w = 0.0
#b = 0.0 

# model prediction 

def forward(x):
    return w*x
    #return w*x+b

# loss = MSE
# MSE = 1/N * (w*x - y)**2

def loss(y,y_hat):
    return ((y_hat-y)**2).mean()



# gradient = 1/N 2x (yhat-y) // calculated mathematically dL/dw

def gradient(x,y,y_hat):
    return np.dot(2*x,(y_hat-y)).mean() 
    #dw = np.dot(2*x,(y_hat-y)).mean()
    #db = (2*(y_hat-y)).mean()
    #return dw,db

initial_prediction = forward(5)

print("initital prediction = " , initial_prediction)


# training 

lr = 0.01
epochs = 16

for epoch in range(epochs):
    #forward pass
    y_pred = forward(X)
    
    #calculating loss
    L = loss(Y,y_pred)
    
    # gradients at each node
    dw = gradient(X,Y,y_pred)
    #dw,db = gradient(X,Y,y_pred)
    
    
    
    #update weights // in gradient descent optimization u update the weight in negative direction of gradient
    w -= lr*dw
    #db = -lr*db

    
    if (epoch%2==0):
        print(f'epoch {epoch} : w = {w:.3f}, loss = {L:.6f}')
    
       
prediction_after_training = forward(5);
    
print("prediction_after_training = ", prediction_after_training)
    
    




