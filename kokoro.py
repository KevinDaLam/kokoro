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
        self._cost = []

        parameters = [np.matrix(np.zeros([hidden_size, input_size + self._bias]))]

        # Adding hidden-to-hidden layer matrices if more than one hidden layer is defined
        for i in range(self._hidden_layers - 1):
            parameters.append(np.matrix(np.zeros([hidden_size, hidden_size + self._bias])))

        parameters.append(np.matrix(np.zeros([output_size, hidden_size + self._bias])))

        self._parameters = parameters

    # Helper for Sigmoid Function
    def Sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # Helper for Sigmoid Deriative Function
    def SigmoidPrime(self, z):
        return np.multiply(self.Sigmoid(z), (1 - self.Sigmoid(z)))

    # Initialize Weights to Random to Allow Neuron Weights to Differ
    def RandomizeWeights(self):

        for i in range(self._hidden_layers + 1):
            self._parameters[i] = np.matrix(np.random.rand(self._parameters[i].shape[0], self._parameters[i].shape[1]))

    def ForwardPropogate(self, input_dataset):

        num_input = input_dataset.shape[0]

        # Add ones to first column to account for bias weight 
        _input_dataset = np.insert(input_dataset, 0, values=np.ones(num_input), axis=1)

        z = []
        a = [_input_dataset]

        for i in range(self._hidden_layers + 1):
            z.append(a[i] * self._parameters[i].T)
            a.append(np.insert(self.Sigmoid(z[i]), 0, values=np.ones(z[i].shape[0]), axis=1))

        a[-1] = np.delete(a[self._hidden_layers+1], 0, axis=1)
        return a, z

    def Cost(self, input_dataset, true_output_data):

        n_training_examples = input_dataset.shape[0]

        a, z = self.ForwardPropogate(input_dataset)

        total_cost = 0
        for i in range(n_training_examples):
            first_term = np.multiply(true_output_data[i,:], np.log(a[-1][i,:]))
            second_term = np.multiply(1 - true_output_data[i,:], np.log(1 - a[-1][i,:]))
            total_cost += np.sum(first_term + second_term)

        # first_term = np.multiply(true_output_data, np.log(a[-1]))
        # second_term = np.multiply(1 - true_output_data, np.log(1 - a[-1]))
        # total_cost = np.sum(first_term + second_term)

        total_cost = -total_cost / n_training_examples
        return total_cost

    def BackPropogate(self, a, z, true_output_data):

        n_training_examples = a[0].shape[0]
        acc_deltas = []
        for i in range(len(self._parameters)):
            acc_deltas.append(np.matrix(np.zeros(self._parameters[i].shape)))

        # Initializing deltas for each layer, deltas[0] is not used
        deltas = [np.matrix(np.zeros((n_training_examples, self._hidden_size))) for x in range(self._hidden_layers)]
        deltas.insert(0,0)

        # To satisfy matrix multiplication given bias units do not have input from previous layers
        for i in range(len(z)-1):
            z[i] = np.insert(z[i], 0, values = np.ones(z[i].shape[0]), axis = 1)

        # Compute last layer delta
        deltas.append(np.multiply(a[-1] - true_output_data, self.SigmoidPrime(z[-1])))

        # Compute delta for last layer
        # print(self._parameters[-1])
        deltas[-2] = np.multiply((self._parameters[-1].T * deltas[-1].T).T, self.SigmoidPrime(z[-2]))

        # Compute delta for preceding layers
        for j in range(self._hidden_layers-1, 0, -1):

            print("PARAMETERS J")
            print(self._parameters[j])
            print("DELTAS J + 1")
            print(deltas[j + 1])
            print("DELTAS J + 1 [:,1:]")
            print(deltas[j + 1][:,1:])
            print("Z MATRIX")
            print(z[j-1])

            deltas[j] = np.multiply((self._parameters[j].T * deltas[j + 1][:,1:].T).T, self.SigmoidPrime(z[j-1]))

            print("DELTAS J")
            print(deltas[j])

        acc_deltas[-1] = (deltas[-1].T * a[-2]) / n_training_examples

        for j in range(len(acc_deltas)-1):
            # print("Delta" + str(j+1))
            # print(deltas[j+1])
            # print("A[%u]" % j)
            # print(a[j])
            acc_deltas[j] = (deltas[j+1][:,1:].T * a[j] ) / n_training_examples
            print(acc_deltas[j])
            # print("ACCUM " + str(j))
            # print(acc_deltas[j])

        # # Regularization
        # for i in range(len(acc_deltas)):
        #     acc_deltas[i][:,1:] = acc_deltas[i][:,1:] + (self._parameters[i][:,1:] * 0.01) / n_training_examples

        for i in range(len(self._parameters)):
            # print("ACCUMULATED DELTAS")
            # print(acc_deltas[i])
            self._parameters[i] = self._parameters[i] - self._learning_rate * acc_deltas[i]

    def Plot_Error_Vs_Training_Epoch(self, n_iterations):

        fig, ax = plt.subplots(figsize=(12,8)) 
        ax.plot(np.arange(n_iterations), self._cost, 'r')  
        ax.set_xlabel('Iterations')  
        ax.set_ylabel('Cost')  
        ax.set_title('Error vs. Training Epoch')
        plt.show()


    def Train(self, input_dataset, output_dataset, n_iterations):

        self.RandomizeWeights()
        self._cost = np.zeros(n_iterations)

        for i in range(n_iterations):
            a, z = self.ForwardPropogate(input_dataset)
            self.BackPropogate(a, z, output_dataset)
            self._cost[i] = (self.Cost(input_dataset, output_dataset))


        self.Plot_Error_Vs_Training_Epoch(n_iterations)

        final_a, final_z = self.ForwardPropogate(input_dataset)
        print(final_a[-1])
        print(output_dataset)
        print("ERROR: ")
        print(abs(np.sum(final_a[-1] - output_dataset)) / output_dataset.shape[0])

    def Predict(self, input_data):

        predicted_a, predicted_z = self.ForwardPropogate(input_data)
        return predicted_a[-1]


if __name__ == '__main__':

    # Test Neural Net with Learning Rate 0.01, Input Vector Size of 2, 2 Hidden Neurons in each Hidden Layer, 2 Hidden Layers, Output Vector Size of 2
    # NN = ANNetwork(0.01, 2, 2, 2, 2)
    # NN.RandomizeWeights()
    # test_input_data = np.matrix([[1,3],[1,2]])
    # test_true_output = np.matrix([[1, 0],[0,1]])
    # a, z = NN.ForwardPropogate(test_input_data)
    # for i in range(len(NN._parameters)):
    #   print("Theta Layer " + str(i))
    #   print(NN._parameters[i])
    # for i in range(len(a)):
    #   print("Activation Layer " + str(i))
    #   print(a[i])
    # NN.Cost(test_input_data, test_true_output)

    # OR Neural Network Test

    input_dataset = np.matrix([
        [1, 0], 
        [1, 1], 
        [0, 0], 
        [0, 1]
    ])
    
    output_dataset = np.matrix([
        [1],
        [1],
        [0],
        [1]
    ])

    XOR_NN = ANNetwork(0.1, 2, 2, 1, 1)
    XOR_NN.Train(input_dataset, output_dataset, 10000)

    p = XOR_NN.Predict(np.matrix([1,0]))
    p2 = XOR_NN.Predict(np.matrix([1,1]))
    p3 = XOR_NN.Predict(np.matrix([0,0]))
    p4 = XOR_NN.Predict(np.matrix([0,1]))
    print(p)
    print(p2)
    print(p3)
    print(p4)
