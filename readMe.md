Basic intro to Pytorch and its functionalities 
We start with this and then study FastAI which is built on top of pytorch and makes training and making models even easier 

FILES : 
1. intro_to_tensors : Basics of tensors, how to convert numpy arrays into tensors, resize, slicing, basic maths operations on tensors etc ...

2. autograd : pytorch's own automatic differentiation library, useful for training gradient based optimization algos, here basics of differentiation were covered

3. backpropogation : basics explained how are derivatives calculated using chain-rule

4. manual_regression : construct a basic linear regression model from scratch without torch - just using numpy and maths.

5. linear_regression_using_pytorch : we replace all manual steps in the previous code by functionalities from pytorch - autograd, optimizers, and linear models.

6. NN_from_scratch : construct a simple NN from scratch without torch on the MNIST data set using numpy and maths to classify digits with 0.85 accuracy

7. dataset & dataloaders : introduction to dataloaders and basic maths involving batches and no of iterations.

8. Custom vs coded dataloaders : Wrote a dataloader from scratch and plot its loading times against the builtin pytorch dataloader to find its actually faster. 
