import numpy as np
x_training, x_testing, y_training, y_testing, b1, w1  = np.array([[0, 0, 1], [1, 1, 1], [1, 0, 1], [0, 1, 1], [1, 0, 0], [1, 1, 0]]).T, np.array([[0, 0, 0], [0, 1, 0]]).T, np.array([[0, 1, 0, 0, 0, 0]]), np.array([[0, 0]]), 0.0, np.random.randn(1, 3)*0.01
for iteration in xrange(60000):
	w1, b1 = w1 - ((0.01/float(x_training.shape[1]))*np.dot(((1 /(1 + np.exp(-(np.dot(w1, x_training) + b1)))) - y_training), x_training.T)), b1 - ((0.01/float(x_training.shape[1]))*np.sum((1 /(1 + np.exp(-(np.dot(w1, x_training) + b1)))) - y_training)) #if i use non log cost function!!!will society accept me?lets not try!!!
print 1 / (1 + np.exp(-(np.dot(w1, x_testing) + b1)))

