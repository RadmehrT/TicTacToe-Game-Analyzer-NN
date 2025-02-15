from UpdatedTicTacToeNN import TicTacToeAI
import matplotlib.pyplot as plt
import numpy as np

neural_network = TicTacToeAI()
    
print("Random synaptic weights: ")
print(neural_network.synaptic_weights)
    
    
filename = 'synaptic weights after training.npz'
    
neural_network.use_previous_weights(filename, 3)
    
#0, 0, 1
expectedAnswers = [0, 0, 1]
    
case1 = neural_network.forward_Propagate(np.array([0,1,1,-1,0,-1,1,-1,0]))
case2 = neural_network.forward_Propagate(np.array([0,1,1,-1,1,1,0,0,0]))
case3 = neural_network.forward_Propagate(np.array([-1,-1,0,0,0,1,1,1,1]))
    
    
    
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
    
ax.bar(categories, expectedAnswers, facecolor='g')
    
plt.xlabel('trial cases')
plt.ylabel('Correct answer')
plt.title("Trial Case Answers")
    
plt.show()