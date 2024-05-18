import torch 
import torch.nn
import torch.nn.functional

y = torch.tensor(2.)
x = torch.tensor(1.)

# forwardpass and compute loss

w = torch.tensor(1., requires_grad=True)
y_hat = w*x
loss = (y_hat-y) ** 2

print(loss)

# backpropogation

loss.backward()
print(w.grad)




