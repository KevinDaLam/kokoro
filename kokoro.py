import numpy as np
import matplotlib.pyplot as plt

class ANNetwork():

	def __init__(self, learning_rate, input_size, hidden_size, hidden_layers, output_size, bias=1):
		self._learning_rate = learning_rate
		self._input_size = input_size
		self._hidden_size = hidden_size
		self._hidden_layers = hidden_layers
		self._output_size = output_size
		self._bias = bias

		parameters = [np.matrix(np.zeros([hidden_size, input_size + self._bias]))]

		# Adding hidden-to-hidden layer matrices if more than one hidden layer is defined
		for i in range(self._hidden_layers - 1):
			parameters.append(np.matrix(np.zeros([hidden_size, hidden_size + self._bias])))

		parameters.append(np.matrix(np.zeros([output_size, hidden_size + self._bias])))

		self._parameters = parameters


	def Sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def SigmoidPrime(self, z):
		return Sigmoid(z) * (1 - Sigmoid(z))

	def RandomizeWeights(self):

		for i in range(self._hidden_layers + 1):
			self._parameters[i] = np.matrix(np.random.rand(self._parameters[i].shape[0], self._parameters[i].shape[1]))

	def Cost(self, data):

		# TODO: CALCULATE COST FUNCTION USING BOTH PREDICTED AND ACTUAL VALUES
		predicted_value, a, z = self.ForwardPropogate()

	def ForwardPropogate(self, input_data):

		num_input = input_data.shape[0]

		# Add ones to first column to account for bias weight 
		input_vector = np.insert(input_data, 0, values=np.ones(num_input), axis=1)

		z = []
		a = [input_vector]

		for i in range(self._hidden_layers + 1):
			z.append(a[i] * self._parameters[i].T)
			a.append(np.insert(self.Sigmoid(z[i]), 0, values=np.ones(z[i].shape[0]), axis=1))

		output_vector = np.delete(a[self._hidden_layers+1], 0, axis=1)

		return output_vector, a, z


	# def BackPropogation(self):



	def Train(self, input_dataset, output_dataset):

		self.RandomizeWeights()



	# def Predict(self, data):

if __name__ == '__main__':

	# Test Neural Net with Learning Rate 0.01, Input Vector Size of 2, 2 Hidden Neurons in each Hidden Layer, 2 Hidden Layers, Output Vector Size of 2
	NN = ANNetwork(0.01, 2, 2, 2, 2)
	NN.RandomizeWeights()
	out, a, z = NN.ForwardPropogate(np.matrix([1,3]))
	print(a)
	print(out)