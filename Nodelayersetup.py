import numpy as np

print("Input number of nodes per layer you want in this format: no1no2no3etc...")
nodelayersetup = input()

layers = len(nodelayersetup)  # Finds total number of layers in the specified multilayered perceptron (MLP).
nodestruct = np.zeros(layers) # Creates an array with the purpose of containing the number of nodes per layer.
elmntsinmlp = 0               # Counter that counts the total number of nodes in MLP.
mlpweights = 0                # Counts weights of all nodes.

for i in range(layers): # len gives total length of string. Loop, loops through the characters and adds them to elmtsinmlp inorder to get the total length of the weight array for the mlp network specified by user.
    elmntsinmlp += int(nodelayersetup[i])
    nodestruct[i] = int(nodelayersetup[i])
    if i != 0: 
        mlpweights += nodestruct[i]*nodestruct[i-1]
    # print(i) remember that the index of arrays in python starts with index 0.

mlpwarr = np.zeros(int(mlpweights)) # Creates array for containing all the weight connections between 

print("Total number of layers: " + str(layers))
print("Total number of nodes: " + str(elmntsinmlp))
print("Total number of weights: " + str(mlpweights))
print("Node structure of MLP: " + str(nodestruct))

