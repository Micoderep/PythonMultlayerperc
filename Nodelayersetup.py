import numpy as np
import math

def sigmoid(x): # Function needs to be before code to be recognised in Python.
    return 1/(1+math.exp(-x))

def feedforward(layers,nodestruct,yarr,mlpwarr):
    presweight = 0 # Keeps track of which weight is to be used in forward passing.
    yarrposi = 0 # Keeps track of position of new set of input nodes.
    yarrposj = 0 # Keeps track of position of old set of input nodes.

    for L in range(layers-1): # Moves between layers denoted by indices 0 ... layer-2.
        jrange = int(nodestruct[L]) # int function as the values are floats otherwise and cannot be used for loop range.
        irange = int(nodestruct[L+1])
        yarrposi += jrange        # Updates position to new set of nodes the information is flowing to.

        for j in range(jrange): # Cycles through the inputs from prior layer. 

            yjpos = yarrposj + j
            for i in range(irange): # Cycles through the nodes of present layer, will be using to cycle through weights.
                yipos = yarrposi + i

                yarr[yipos] += mlpwarr[presweight]*yarr[yjpos]

                presweight += 1 # Keeps incrementing the weight index.



        for i in range(irange): # Applies sigmoid function to the sum of the inputs for new output.
            yipos = yarrposi + i
            yarr[yipos] = sigmoid(yarr[yipos])


        yarrposj += jrange        # Updates positon of old nodes.

    return yarr


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


mlpwarr = np.array([0.4,0.1,0.5,0.2,-0.2,0.8,0.2,-0.9,0.1]) # For example weights system.

# Forward passing information part:

yarr = np.zeros(int(elmntsinmlp)) # Array that contains all the information of the output of nodes, including the intial input.

#yarr[0] = 2.0 # Test inputs.
#yarr[1] = 1.0

yinputs = np.array([0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0]) # Complete set of inputs.
yactualout = np.array([0.0,1.0,1.0,1.0]) # Complete set of actual outputs.

inputnum = int(nodestruct[0]) # number of inputs

trialnum = int(len(yinputs)/inputnum) # Number of datasets/trials to be forwarded through the network.

totalloss = 0

outputs = int(nodestruct[layers-1]) # number of outputs in final layer.
for trial in range(trialnum): # loops through different inputs

    for y in range(inputnum): # Inserts correct inputs corresponding to correct trial
        yinputpos = trial*inputnum + y # Calculates correct position of input in yinputs array.
        yarr[y] = yinputs[yinputpos]

    print(yarr)
    for ifin in range(outputs): # loops through the final layers nodes.
        yarr = feedforward(layers,nodestruct,yarr,mlpwarr) # output corresponding to particular input.

        totalloss += (yarr[ifin-outputs] - yactualout[ifin+trial])*(yarr[ifin-outputs] - yactualout[ifin+trial])
        #Had a problem that was caused by not subtracting the correct actual output from network value.
        #This was solved by adding trial to the index of yactualout because the required element is in a place that depends on the trial number.

    yarr = np.zeros(int(elmntsinmlp)) # resets all elements in the input/output nodes to 0 in order to not let independent trials effect oneanother. Problem was due to me setting yarr = 0 made yarr into an integer rather than setting all elements to 0.


totalloss = totalloss/trialnum


print("Final value of input: " + str(totalloss))



