import numpy as np
import matplotlib.pyplot as plt

# Create input and output data
#x = np.array([ [0, 0], [0, 1], [1, 0]]).T
#y = np.array([[ 0, 1, 1]])
x = np.array([ [0, 0, 0], [0, 1, 0], [1, 0, 0], [0, 1, 1], [1, 1, 1], [1, 1, 0]]).T
y = np.array([[ 0, 1, 1, 0, 1, 0]])


#x = np.array([[5, 4], [3, 11], [6, 7], [1, 9], [0, 10], [10, 0], [10, 10]]).T
#y = np.array([[0, 1, 1, 0, 0, 0, 1]])
HL1 = 10
HL2 = 10

print "input and output", x.shape, y.shape
# Randomly initialize weights
np.random.seed(5)
w1 = np.random.randn(HL1, x.shape[0])*0.01
w2 = np.random.randn(HL2, HL1)*0.01
w3 = np.random.randn(y.shape[0], HL2)*0.01

b1 = np.zeros((HL1, 1))
b2 = np.zeros((HL1, 1))
b3 = np.zeros((y.shape[0], 1))

#print "w1:", w1
#print "w2", w2

learning_rate = 0.1
loss_list = []

for t in range(200000):

    #forward propagation to compute y

    #start of layer 1
    z1 = w1.dot(x) + b1
    
    #ReLU as activate function
    #a1 = np.maximum(z1, 0)  
    
    #sigmoid as activation function
    a1 = 1/(1 + np.exp(-z1))
    
    
    #start of layer 2
    z2 = w2.dot(a1) + b2
    a2 = 1/(1 + np.exp(-z2))
    
    
    #start of layer 3
    z3 = w3.dot(a2) + b3
    y_predicted = 1/(1 + np.exp(-z3))
    
    
    #print "z1 and its shape:", z1, z1.shape,"\n", "a1 and its shape", a1, a1.shape,"\n", "y_predicted and it shape", y_predicted, y_predicted.shape	    
    
    #compute the loss
    loss = np.square(y_predicted - y).sum() # loss function
    loss_list.append(loss)
    #print(loss)    
    
    #Backpropagate error to compute gradients of w1 and w2 with respect to loss
    
    #the last layer's error's derivative
    
    dy = 2.0 * (y_predicted - y) 
    #print "dy", dy
    dz3 = dy*y_predicted*(1-y_predicted)
    dw3 = dz3.dot(a2.T)
    db3 = np.sum(dz3, axis=1, keepdims=True)
    #second layer's error's derivative
    da2 = w3.T.dot(dz3)
    #we have used relu function so values will be same if greater than 0
    #dz1 = da1.copy()
    #dz1[z1 < 0] = 0
    
    #gradient for sigmoid function
    dz2 = da2*a2*(1 - a2)
    dw2 = dz2.dot(a1.T)
    db2 = np.sum(dz2, axis=1, keepdims=True)
    #first layer's error's derivative
    da1 = w2.T.dot(dz2)
    dz1 = da1*a1*(1-a1)
    dw1 = dz1.dot(x.T)
    db1 = np.sum(dz1, axis=1, keepdims=True)

    #update weights
    w1 = w1 - learning_rate * dw1
    w2 = w2 - learning_rate * dw2
    w3 = w3 - learning_rate * dw3
    
    b1 = b1 - learning_rate * db1
    b2 = b2 - learning_rate * db2
    b3 = b3 - learning_rate * db3
    #break
print "w1 and w2 are:", w1, w2
print "predicted output is:", 1/(1+np.exp(-w3.dot(1/(1+np.exp(-w2.dot(1/(1+np.exp(-w1.dot(x)-b1)))-b2)))-b3))
print "predicted output is:", 1/(1+np.exp(-w3.dot(1/(1+np.exp(-w2.dot(1/(1+np.exp(-w1.dot(np.array([[1, 0, 1]]).T)- b1)))- b2)))- b3))
plt.plot(loss_list)
plt.show()
