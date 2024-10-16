import numpy as np
import math

def sigmoid(x): # Function needs to be before code to be recognised in Python.
    return 1/(1+math.exp(-x))

def feedforward(layers,nodestruct,yarr,mlpwarr,trial,elmntsinmlp,mlpweights):
    presweight = 0+trial*int(mlpweights) # Keeps track of which weight is to be used in forward passing.
    yarrposi = 0 # Keeps track of position of new set of input nodes.
    yarrposj = 0 # Keeps track of position of old set of input nodes.

    for L in range(layers-1): # Moves between layers denoted by indices 0 ... layer-2.
        jrange = int(nodestruct[L]) # int function as the values are floats otherwise and cannot be used for loop range.
        irange = int(nodestruct[L+1])
        yarrposi += jrange        # Updates position to new set of nodes the information is flowing to.

        for j in range(jrange): # Cycles through the inputs from prior layer. 

            yjpos = yarrposj + j + elmntsinmlp*trial
            for i in range(irange): # Cycles through the nodes of present layer, will be using to cycle through weights.
                yipos = yarrposi + i + elmntsinmlp*trial

                #print("presweight: " + str(presweight))
                #print("trial: " + str(trial))
                yarr[yipos] += mlpwarr[presweight]*yarr[yjpos]
                 
                presweight += 1 # Keeps incrementing the weight index.



        for i in range(irange): # Applies sigmoid function to the sum of the inputs for new output.
            yipos = yarrposi + i + elmntsinmlp*trial
            yarr[yipos] = sigmoid(yarr[yipos])


        yarrposj += jrange        # Updates positon of old nodes.

    return yarr[elmntsinmlp*trial:elmntsinmlp*(trial+1)]


def trialruns(trialnum,inputnum,yarr,yinputs,elmntsinmlp,outputs,layers,nodestruct,mlpwarr,mlpweights):
    totalloss = 0
    for trial in range(trialnum): # loops through different inputs

        for y in range(inputnum): # Inserts correct inputs corresponding to correct trial
            yinputpos = trial*inputnum + y # Calculates correct position of input in yinputs array.
            yarrpos = y + elmntsinmlp*trial
            yarr[yarrpos] = yinputs[yinputpos]

        yarrposstart = elmntsinmlp*trial
        yarr[yarrposstart:yarrposstart+elmntsinmlp] = feedforward(layers,nodestruct,yarr,mlpwarr,trial,elmntsinmlp,mlpweights) # output corresponding to particular input.
        for ifin in range(outputs): # loops through the final layers nodes.
            
            tlsyarrpos = elmntsinmlp*(trial+1)-outputs+ifin
            totalloss += (yarr[tlsyarrpos] - yactualout[ifin+trial])*(yarr[tlsyarrpos] - yactualout[ifin+trial])
        #Had a problem that was caused by not subtracting the correct actual output from network value.
        #This was solved by adding trial to the index of yactualout because the required element is in a place that depends on the trial number.

    return totalloss/trialnum, yarr


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
mlpwarr = np.append(mlpwarr,[0.4,0.1,0.5,0.2,-0.2,0.8,0.2,-0.9,0.1]) # For example weights system.
mlpwarr = np.append(mlpwarr,[0.4,0.1,0.5,0.2,-0.2,0.8,0.2,-0.9,0.1]) # For example weights system.
mlpwarr = np.append(mlpwarr,[0.4,0.1,0.5,0.2,-0.2,0.8,0.2,-0.9,0.1]) # For example weights system.
# Forward passing information part:

yinputs = np.array([0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0]) # Complete set of inputs.
yactualout = np.array([0.0,1.0,1.0,1.0]) # Complete set of actual outputs.

inputnum = int(nodestruct[0]) # number of inputs

trialnum = int(len(yinputs)/inputnum) # Number of datasets/trials to be forwarded through the network.

yarr = np.zeros(int(elmntsinmlp*trialnum)) # Array that contains all the information of the output of nodes, including the intial input.
outputs = int(nodestruct[layers-1]) # number of outputs in final layer.

totalloss, yarr = trialruns(trialnum,inputnum,yarr,yinputs,elmntsinmlp,outputs,layers,nodestruct,mlpwarr,mlpweights)
print(yarr)
print("Now weights")
print(mlpwarr)

print("Final value of input: " + str(totalloss))


#Now to do the back propagation after getting an array containing all outputs (yarr).
#Last layer first:

dsdw = 0 # Change in the loss function for a change in weight.
lp = 1 # Learning parameter.

# Here is where the first layer loop is ***********************
for ifin in range(outputs): # For if you had multiple outputs.
    for nodej in range(int(nodestruct[layers-2])): # Loop for second to last layer!

        presweight = -1*(nodej+1)
        for trial in range(trialnum): # loops over all trials for 

            yinodepos = elmntsinmlp*(trialnum-trial)-ifin-1 # Starts from the end of each set and works up
            yacpos = len(yactualout)-trial-1-ifin
            yjnodepos = elmntsinmlp*(trialnum-trial)-nodej-outputs-1 # position of the correct prior layer node corresponding to the weight currently being changed.
            #print("position: " + str(position))
            #print("yacpos: " + str(yacpos))
            #print("yjnodepos: " + str(yjnodepos))
            yval = yarr[yinodepos]
            dsdw = (yval-yactualout[yacpos])*yval*(1-yval)*yarr[yjnodepos]
            print((yval-yactualout[yacpos])*yval*(1-yval)*yarr[yjnodepos])
            print(trialnum)
            print("trial: " + str(trial))
            print("mlpwbefore: " + str(mlpwarr[presweight]))
            mlpwarr[presweight] = mlpwarr[presweight] - (lp*dsdw)
            print("mlpwafter: " + str(mlpwarr[presweight]) + " Weight: " + str(presweight))

            presweight = -1*(nodej+1)-(trial+1)*int(mlpweights)
            dsdw = 0


# More layers:


totalloss, yarr = trialruns(trialnum,inputnum,yarr,yinputs,elmntsinmlp,outputs,layers,nodestruct,mlpwarr,mlpweights)
print(yarr)
print("Now weights")
print(mlpwarr)
print("Final value of input: " + str(totalloss))
