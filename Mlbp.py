import numpy as np
import math

def sigmoid(x): # Function needs to be before code to be recognised in Python.
    return 1/(1+np.exp(-x))

def feedforward(layers,nodestruct,yarr,mlpwarr,trial,elmntsinmlp,mlpweights):
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

                #print("presweight: " + str(presweight))
                #print("trial: " + str(trial))
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

yinputs = np.array([0.0,0.0,0.0,1.0,1.0,0.0,1.0,1.0]) # Complete set of inputs.
yactualout = np.array([0.0,1.0,1.0,1.0]) # Complete set of actual outputs.

inputnum = int(nodestruct[0]) # number of inputs

trialnum = int(len(yinputs)/inputnum) # Number of datasets/trials to be forwarded through the network.

yarr = np.zeros(elmntsinmlp) # Array that contains all the information of the output of nodes, including the intial input.
outputs = int(nodestruct[layers-1]) # number of outputs in final layer.
deltal = np.zeros(elmntsinmlp) # Array that contains the deltal corresponding to a particular node in a particular layer.

totalloss = 0
trial = 0 # so I dont have to delete all my code and because I am going to make it go through different trials after completing one.


for y in range(inputnum): # Inserts correct inputs corresponding to correct trial
    yinputpos = trial*inputnum + y # Calculates correct position of input in yinputs array.
    yarrpos = y
    yarr[yarrpos] = yinputs[yinputpos]

yarr = feedforward(layers,nodestruct,yarr,mlpwarr,trial,elmntsinmlp,mlpweights) # output corresponding to particular input.

for ifin in range(outputs): # loops through the final layers nodes.

    tlsyarrpos = elmntsinmlp-ifin-1
    print((trial+1)*outputs-ifin-1)
    totalloss += (yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])*(yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])
        #Had a problem that was caused by not subtracting the correct actual output from network value.
        #This was solved by adding trial to the index of yactualout because the required element is in a place that depends on the trial number.

print("totalloss: " + str(totalloss))
print("yarr: ")
print(yarr)
print("mlpwarr: ")
print(mlpwarr)
    
dsdw = 0 # Change in the loss function for a change in weight.
lp = 1 # Learning parameter.

# Here is where the first layer loop is ***********************
for ifin in range(outputs): # For if you had multiple outputs.
    yacpos = (trial+1)*outputs-ifin-1
    yarrpos = elmntsinmlp-1-ifin
    print("ifin: " + str(ifin) + " yacpos: " + str(yacpos) + " yarrpos: " + str(yarrpos))
    yval = yarr[yarrpos]
    deltal[yarrpos] = (yval-yactualout[yacpos])*yval*(1-yval) # Assigns deltal to array in spot corresponding to the output node.

#print(deltal)
weightpos = 0
for lm1 in range(int(nodestruct[layers-2])):
    for ifin in range(outputs):
        dpos =  elmntsinmlp-1-ifin
        yarrpos = elmntsinmlp-outputs-1-lm1
        weightpos += -1

        dsdw = deltal[dpos]*yarr[yarrpos]
        mlpwarr[weightpos] = mlpwarr[weightpos] - lp*dsdw

        #print("lm1: " + str(lm1) + " ifin: " + str(ifin) + " dpos: " + str(dpos) + " yarrpos: " + str(yarrpos) + " weightpos: " + str(weightpos) + " dsdw: " + str(dsdw)) 

#print(mlpwarr)

####################################### For all other layers:
istart = -int(nodestruct[layers-1])
dlstrt = 0
jstart = -int(nodestruct[layers-1])
weightstart = 0
weightstart2 = 0

for L in range(layers-2): #-2 as the first iteration has already been done above and there are no weights before the first layer.
    weightstart += -int(nodestruct[L-1]*nodestruct[L-2])

    jstart += -int(nodestruct[layers-L-2])
    ###############################
    # Calculates the new dl values from layer change
    for dl in range(int(nodestruct[layers-2-L])): # Adds weighted dl's in previous layer to make new ones
        # Cycles through different dl values adds the ones from different nodes in the previous layer.
                
        deltapoint2 = -1+istart-dl
        for di in range(int(nodestruct[layers-1-L])): # Cycles through nodes in previous layer
            mlpwpoint = weightstart2 -dl-1-di*int(nodestruct[layers-2-L]) # The -1 may mess with calculations
            deltapoint = -dlstrt-di-1

            deltal[deltapoint2] += mlpwarr[mlpwpoint]*deltal[deltapoint]
                 
        deltal[deltapoint2] = deltal[deltapoint2]*yarr[deltapoint2]*(1-yarr[deltapoint2]) 

    #############################
    # Applies new dl values and calculates the dsdw values.
    for nodej in range(int(nodestruct[layers-3-L])): # For if you had multiple outputs.

        for dl in range(int(nodestruct[layers-2-L])): # Adds weighted dl's in previous layer to make new ones
        # Cycles through different dl values adds the ones from different nodes in the previous layer.

            weightpoint = weightstart - dl - nodej*int(nodestruct[layers-2-L]) - 1
            yarrpoint = jstart - nodej - 1
            deltapoint = istart - dl - 1
            dsdw = deltal[deltapoint]*yarr[yarrpoint]
            mlpwarr[weightpoint] = mlpwarr[weightpoint] - (dsdw*lp)

    dlstrt += int(nodestruct[layers-1-L]) # alters the layer at which the dl values are calculated for.
    weightstart2 += -int(nodestruct[L-1]*nodestruct[L-2])
    istart += -int(nodestruct[layers-L-2])

print("mlpwarr: ")
print(mlpwarr)

yarr = np.zeros(elmntsinmlp)

yarr = feedforward(layers,nodestruct,yarr,mlpwarr,trial,elmntsinmlp,mlpweights) # output corresponding to particular input.

print("yarr: ")
print(yarr)

print("deltal: ")
print(deltal)
totalloss = 0
for ifin in range(outputs): # loops through the final layers nodes.

    tlsyarrpos = elmntsinmlp-ifin-1
    print((trial+1)*outputs-ifin-1)
    print(yarr[tlsyarrpos])
    print(yactualout[(trial+1)*outputs-ifin-1])
    totalloss += (yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])*(yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])
        #Had a problem that was caused by not subtracting the correct actual output from network value.
        #This was solved by adding trial to the index of yactualout because the required element is in a place that depends on the trial number.

print("totalloss: " + str(totalloss))
