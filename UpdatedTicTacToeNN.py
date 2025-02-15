import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from TicTacToeDataProcessor import TicTacToeDataProcessor


class TicTacToeAI():
    def __init__(self):
        np.random.seed(1)
        
        inputH1Size = 7
        h1Size = 5
        
        weights_input_hidden1 = 2 * np.random.random((9, inputH1Size)) * np.sqrt(1 / inputH1Size)
        
        weights_hidden1 = 2 * np.random.random((inputH1Size, h1Size)) * np.sqrt(1 / h1Size)
        
        weights_hidden1_output = 2 * np.random.random((h1Size, 1))

        self.synaptic_weights = [weights_input_hidden1, weights_hidden1, weights_hidden1_output]
        
        self.biases = [np.zeros((1, inputH1Size)), np.zeros((1, h1Size)), np.zeros((1, 1))]
    
    def use_previous_weights(self, filename, array_amount):
        loaded_data = np.load(filename)
        
        self.synaptic_weights = []
        
        for i in range(array_amount):
            i += 1
            self.synaptic_weights.append(loaded_data['array' + str(i)])
        

    
    def tanh(self, x):
        x = np.clip(x, -500, 500) #Clipping x to avoid runtime overflow
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1.0 - np.tanh(x) ** 2
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def forward_Propagate(self, inputs):
        self.layer1_output = self.tanh(np.dot(inputs, self.synaptic_weights[0]) + self.biases[0])  
        
        self.layer2_output = self.tanh(np.dot(self.layer1_output, self.synaptic_weights[1]) + self.biases[1])
        
        output = self.sigmoid(np.dot(self.layer2_output, self.synaptic_weights[2]) + self.biases[2]) 
        
        return output
    
    def train(self, training_inputs, training_outputs, training_iterations, learning_rate):
        
        print("Training inputs shape:", training_inputs.shape)
        
        data_size = training_inputs.shape[0]
        
        for iteration in range(training_iterations):
            
            batch_size = 128
            
            permutation = np.random.permutation(data_size)
            training_inputs = training_inputs[permutation]
            training_outputs = training_outputs[permutation]
            
            
            for i in range(0, batch_size):
                
                end = min(i + batch_size, data_size)
            
                
                batch_inputs = training_inputs[i:end]
                batch_outputs = training_outputs[i:end]

                batch_inputs = batch_inputs.reshape(-1, 9)  
                batch_outputs = batch_outputs.reshape(-1, 1)  
                
                output = self.forward_Propagate(batch_inputs)

                
                error = batch_outputs - output
                
                
                # updates weights in reverse order (The first one updated is the output and the last one is the input)
                
                delta_output = error * self.sigmoid_derivative(output)
                adjustments_output = np.dot(self.layer2_output.T, delta_output)  
                
                error_hidden1 = np.dot(delta_output, self.synaptic_weights[2].T)  
                delta_hidden1 = error_hidden1 * self.tanh_derivative(self.layer2_output) 
                adjustments_hidden1 = np.dot(self.layer1_output.T, delta_hidden1)  
                
                error_input = np.dot(delta_hidden1, self.synaptic_weights[1].T)
                delta_input = error_input * self.tanh_derivative(self.layer1_output)
                adjustments_inputs = np.dot(batch_inputs.T, delta_input)

                self.synaptic_weights[2] += adjustments_output * learning_rate
                self.synaptic_weights[1] += adjustments_hidden1 * learning_rate
                self.synaptic_weights[0] += adjustments_inputs * learning_rate
                
                self.biases[2] += learning_rate * np.sum(delta_output, axis=0)
                self.biases[1] += learning_rate * np.sum(delta_hidden1, axis=0)
                self.biases[0] += learning_rate * np.sum(delta_input, axis=0)
                
                print("Running Iteration: ", iteration, "On input: ", i) 


if __name__ == "__main__":
    neural_network = TicTacToeAI()
    
    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)
    
    file_path = "converted_Data.csv" 
    processor = TicTacToeDataProcessor(file_path=file_path)

    processor.load_data()
    processor.shuffle_data(5)

    training_inputs = processor.get_training_inputs()

    training_outputs = processor.get_training_outputs().T  
    
    filename = 'synaptic weights after training.npz'
    
    if str(input("Use previous training weights? y/n \n")) == "y" :
        neural_network.use_previous_weights(filename, 3)
    
    if str(input("Delete previous traning weights save file? y/n \n")) == "y": 
        if os.path.exists(filename):
            os.remove(filename)
    
    neural_network.train(training_inputs, training_outputs, 2000, 0.01)
    
    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)
    
    dataToSave = [neural_network.synaptic_weights[0], neural_network.synaptic_weights[1], neural_network.synaptic_weights[2]]
    
    np.savez('synaptic weights after training.npz', array1=dataToSave[0], array2=dataToSave[1], array3=dataToSave[2])
    
    case1 = neural_network.forward_Propagate(np.array([1,1,1,1,0,0,0,0,1]))
    case2 = neural_network.forward_Propagate(np.array([0,1,0,1,1,1,1,0,0]))
    case3 = neural_network.forward_Propagate(np.array([0,0,-1,1,0,1,1,0,1]))
    
    
    
    print(case1)
    print(case2)
    print(case3)
    
    fig, ax = plt.subplots(layout='constrained')
    categories = ["test 1", "test 2,", "test 3"]
    
    ax.bar(categories, [float(case1), float(case2), float(case3)])
    
    plt.xlabel('trial cases')
    plt.ylabel('output of NN')
    plt.title("Results of NN")
    
    #Second graph

    fig, ax = plt.subplots(layout='constrained')
    categories = ["test 1", "test 2,", "test 3"]
    
    ax.bar(categories, [1, 1, 0], facecolor='g')
    
    plt.xlabel('trial cases')
    plt.ylabel('Correct answer')
    plt.title("Trial Case Answers")
    
    plt.show()