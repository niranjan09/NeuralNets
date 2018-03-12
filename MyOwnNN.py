import numpy as np
from scipy.special import expit


class networkLayer:
    def __init__(self, nodes_count, prev_layer_nodes_count):
        self.nwm = np.random.rand(nodes_count, prev_layer_nodes_count)
        self.layer_bias = np.random.rand(nodes_count)

    def compute_layer(self, layer_input_matrix):
        #layer_input_matrix is assumed to be 1xn matrix
        return expit(layer_input_matrix.dot(self.nwm) + self.layer_bias)

    def update_layer_weights(self, weight_matrix):
        self.nwm = weight_matrix

class neuralNetwork:
    def __init__(self, train_data_input, train_data_output, possible_outputs, total_no_of_hidden_layers = 1,architecture = [], learning_rate = 0.01, epoch = 100):
        self.tdi = train_data_input
        self.tdo = train_data_output
        self.possible_outputs = possible_outputs
        self.nohl = total_no_of_hidden_layers
        self.learning_rate = learning_rate
        if architecture==[]:
            self.arch = [len(self.tdi)]*self.nohl
        else:
            self.arch = architecture
        self.networkLayersList = []
        self.input_layer = networkLayer(len(self.tdi[0]))
        #networkLayersList.append(input_layer)
        for layer_no in range(self.nohl):
            self.networkLayersList.append(networkLayer(architecture[layer_no]))
        self.output_layer = networkLayer(len(self.possible_outputs))
        #networkLayersList.append(output_layer)

    def train_neural_network():
        op_of_prop = self.propagate_forward()
        self.propagate_backward((self.tdo - op_of_prop)**2)



    def propagate_forward(self):
        op_of_layer = self.input_layer.compute_layer(self.tdi)
        for nw_layer in self.networkLayersList:
            op_of_layer = nw_layer.compute_layer(op_of_layer)
        return self.output_layer.compute_layer(op_of_layer)


    def propagate_backward(self, output_error):
        next_layer_error = output_error
        #change weights of output layer
        new_weights = derivative(self.output_layer.nwm) * learning_rate
        self.output_layer.update_layer_weights(new_weights)
        for nw_layer in self.networkLayersList[::-1]:
