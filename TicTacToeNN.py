import numpy as np
import pandas as pd
from TicTacToeDataProcessor import TicTacToeDataProcessor

class TicTacToeAI():
    def __init__(self):
        np.random.seed(1)
        
        weights_input_hidden1 = 2 * np.random.random((9, 9))
        
        weights_hidden1 = 2 * np.random.random((9, 3)) 
        
        weights_hidden1_output = 2 * np.random.random((3, 1)) 

        self.synaptic_weights = [weights_input_hidden1, weights_hidden1, weights_hidden1_output]
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def forward_Propogate(self, inputs):
        self.layer1_output = self.sigmoid(np.dot(inputs, self.synaptic_weights[0]))  
        
        self.layer2_output = self.sigmoid(np.dot(self.layer1_output, self.synaptic_weights[1]))
        
        output = self.sigmoid(np.dot(self.layer2_output, self.synaptic_weights[2])) 
        return output
    
    def train(self, training_inputs, training_outputs, training_iterations):
        
        print("Training inputs shape:", training_inputs.shape)
        
        data_size = training_inputs.shape[0]
        
        for iteration in range(training_iterations):
            
            batch_size = 128
            
            permutation = np.random.permutation(data_size)
            training_inputs = training_inputs[permutation]
            training_outputs = training_outputs[permutation]
            
            
            for i in range(0, data_size, batch_size):
                
                end = min(i + batch_size, data_size)
            
                
                batch_inputs = training_inputs[i:end]
                batch_outputs = training_outputs[i:end]

                batch_inputs = batch_inputs.reshape(-1, 9)  
                batch_outputs = batch_outputs.reshape(-1, 1)  
                
                output = self.forward_Propogate(batch_inputs)

                
                error = batch_outputs - output
                
                
                # updates weights in reverse order (The first one updated is the output and the last one is the input)
                
                delta_output = error * self.sigmoid_derivative(output)
                adjustments_output = np.dot(self.layer2_output.T, delta_output)  
                
                error_hidden1 = np.dot(delta_output, self.synaptic_weights[2].T)  
                delta_hidden1 = error_hidden1 * self.sigmoid_derivative(self.layer2_output) 
                adjustments_hidden1 = np.dot(self.layer1_output.T, delta_hidden1)  
                
                error_input = np.dot(delta_hidden1, self.synaptic_weights[1].T)
                delta_input = error_input * self.sigmoid_derivative(self.layer1_output)
                adjustments_inputs = np.dot(batch_inputs.T, delta_input)

                self.synaptic_weights[2] += adjustments_output
                self.synaptic_weights[1] += adjustments_hidden1
                self.synaptic_weights[0] += adjustments_inputs
                
                print("Running Iteration: ", iteration, "On input: ", i) 


if __name__ == "__main__":
    neural_network = TicTacToeAI()
    
    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)
    
    file_path = "C:/Users/radme/Desktop/Python Test/Neural Network stuff/TicTacToeAI/converted_Data.csv" 
    processor = TicTacToeDataProcessor(file_path=file_path)

    processor.load_data()

    training_inputs = processor.get_training_inputs()

    training_outputs = processor.get_training_outputs().T  
    
    neural_network.train(training_inputs, training_outputs, 1000)
    
    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)
    
    print(neural_network.forward_Propogate(np.array([1,1,1,1,0,0,0,0,1])))
    print(neural_network.forward_Propogate(np.array([0,1,0,1,1,1,1,0,0])))
    print(neural_network.forward_Propogate(np.array([0,0,-1,1,0,1,1,0,1])))
