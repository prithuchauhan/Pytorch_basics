import torch
import torch.nn as nn
import torch.nn.functional
import numpy as np
from sklearn import datasets        # to make a regression dataset
import matplotlib.pyplot as plt


# three steps 
# 1. Model design (input features, output size, forward pass)
# 2. construct loss and optim
# 3. Training loop 
        # forward pass 
        # backward pass
        # update weights
        

X_numpy, Y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1) 

#convert to tensor
X = torch.from_numpy(X_numpy.astype(np.float32))
Y = torch.from_numpy(Y_numpy.astype(np.float32))
Y = Y.reshape(100,1)



n_samples, n_features = X.shape

# now we have the data make the model

# 1. MODEL 

input_size = n_features  # yeh dekhlo features = input_size tells us the no of features like size, age, location, in iris data set for example petal data length and stuff 
output_size = 1     # one prediction value 
model = nn.Linear(input_size, output_size)

# 2. Loss and optimizer

learningrate = 0.01
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learningrate)   # specify learning rate here or anywhere else # sgd is stochastic gradient descent

num_epochs = 100

for epoch in range(num_epochs):
    
    #forward pass and loss
    
    y_pred = model(X)
    loss = criterion(y_pred, Y)
    
    # backward pass
    
    loss.backward()  # calculates gradients for us automatically
    
    # update weights
    
    optimizer.step()
    
    # change gradients to zero.  why ??? 
    
    optimizer.zero_grad()
    
    
    if (epoch%10 == 0):
        print(f'epoch = {epoch} & loss = {loss:.4f}')
    
predicted = model(X).detach().numpy()      # this predicted is a tensor of all y_pred for the given 100 x. 
 
# cant call numpy on a tensor that requires grad hence detach is used.


# plot it 

plt.plot(X_numpy,Y_numpy,'ro')
plt.plot(X_numpy,predicted,'b')
plt.legend()
plt.show()