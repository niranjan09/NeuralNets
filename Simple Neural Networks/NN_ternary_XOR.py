#ternary xor test ...check pass or fail by running this code
#nn is not powerful without depth?
import numpy as np

np.random.seed(2)
X = np.array([ [1,1,1],[0,1,1],[1,0,1],[0,0,0],[1,1,0] ]).T
Y = np.array([ [1,0,0,0,0] ])
print "The input X is:", X, X.shape
print "The expected output Y is:", Y, Y.shape

input_layer_size = X.shape[0]
hidden_layers_list = [5]
output_layer_size = Y.shape[0]
layers_list = [input_layer_size] + hidden_layers_list + [output_layer_size]
print(layers_list)

parameters = {}
parameters['epoch'] = 5
parameters['alpha'] = 0.01
cache = {}
cache['A0'] = X



#Random initialization of weights and bias values of networks
for layer in range(1, len(layers_list)):
	parameters["W"+str(layer)] = np.random.randn(layers_list[layer], layers_list[layer-1])*0.1
	parameters["b"+str(layer)] = np.zeros((layers_list[layer], 1))
	print("The randomly initialized values are:", parameters['W'+str(layer)], parameters['W'+str(layer)].shape, parameters['b'+str(layer)], parameters['b'+str(layer)].shape)



for run in range(parameters['epoch']):
	
	#forward propagation for first l-1 layers:
	for layer in range(1, len(layers_list)-1):
		cache['Z'+str(layer)] = parameters["W"+str(layer)].dot(cache['A'+str(layer-1)]) + parameters["b"+str(layer)]
		print 'Z'+str(layer), cache['Z'+str(layer)]
		cache['A'+str(layer)] = 1/(1 + np.exp(-cache['Z'+str(layer)]))
		print 'A'+str(layer) , cache['A'+str(layer)]
	
	
	#implementing last layer; keeping this layer different from other layers to use different functions here in future:
	cache['Z'+str(len(layers_list)-1)] = parameters["W"+str(len(layers_list)-1)].dot(cache['A'+str(len(layers_list)-2)]) + parameters["b"+str(len(layers_list)-1)]
	print 'Z'+str(len(layers_list)-1), cache['Z'+str(len(layers_list)-1)]
	cache['A'+str(len(layers_list)-1)] = 1/(1 + np.exp(-cache['Z'+str(len(layers_list)-1)]))
	print 'A'+str(len(layers_list)-1) , cache['A'+str(len(layers_list)-1)]
	
	
	#print (Y - cache['A'+str(len(layers_list)-1)])**2
	#compute cost
	cache['E'+str(len(layers_list)-1)] = (Y - cache['A'+str(len(layers_list)-1)])
	print 'E'+str(len(layers_list)-1), cache['E'+str(len(layers_list)-1)]
	
	
	
	
	#backpropagate error
	for layer in range(len(layers_list)-2, 0, -1):
		cache['E'+str(layer)] = (parameters['W'+str(layer+1)].T).dot(cache['E'+str(layer+1)])
		print 'E'+str(layer), ':', cache['E'+str(layer)]
	
	
	
	
	
	#calculate dw with the help of error
	for layer in range(1, len(layers_list)-1):
		#calculate derivative here:
		cache['dW'+str(layer)] = (-(cache['E'+str(layer)])*(cache['A'+str(layer)]*(1-cache['A'+str(layer)]))).dot(cache['A'+str(layer-1)].T)
		parameters['W'+str(layer)] = parameters['W'+str(layer)] - parameters['alpha'] * (cache['dW'+str(layer)])
		print parameters['W'+str(layer)]
		
	cache['dW'+str(len(layers_list)-1)] = (-(cache['E'+str(len(layers_list)-1)])*(cache['A'+str(len(layers_list)-1)]*(1-cache['A'+str(len(layers_list)-1)]))).dot(cache['A'+str(len(layers_list)-2)].T)
	parameters['W'+str(len(layers_list)-1)] = parameters['W'+str(len(layers_list)-1)] - parameters['alpha'] * (cache['dW'+str(len(layers_list)-1)])
	print parameters['W'+str(len(layers_list)-1)]
	
	#break
	
	
print cache['A2']



a = np.array([[1, 1, 1]]).T
for layer in range(1, len(layers_list)):
	z = parameters["W"+str(layer)].dot(a) + parameters["b"+str(layer)]
	a = 1/(1 + np.exp(-z))

print a
	
