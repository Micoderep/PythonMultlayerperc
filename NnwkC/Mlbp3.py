import numpy as np
import math
import sys

def sigmoid(x): # Function needs to be before code to be recognised in Python.
    return 1/(1+np.exp(-x))

def feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights):
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

############################################################# Start of program:

with open("mlpstruct.txt", "r") as filest: # This code reads in the number of nodes for a particular layer of the neural network.
    f_list = [int(i) for line in filest for i in line.split(' ') if i.strip()]
    nodestruct = f_list

filest.close()

layers = len(nodestruct) # Extracts the number of layers from the length of nodestruct.

elmntsinmlp = 0               # Counter that counts the total number of nodes in MLP.
mlpweights = 0                # Counts weights of all nodes.

for i in range(layers): # Loop, loops through the characters and adds them to elmtsinmlp inorder to get the total length of the weight array for the mlp network specified by user.
    elmntsinmlp += nodestruct[i]
    if i != 0: 
        mlpweights += nodestruct[i]*nodestruct[i-1]
    # Remember that the index of arrays in python starts with index 0.

mlpweights = int(mlpweights)
mlpwarr = np.zeros(int(mlpweights)) # Creates array for containing all the weight connections between 

print("Total number of layers: " + str(layers))
print("Total number of nodes: " + str(elmntsinmlp))
print("Total number of weights: " + str(mlpweights))
print("Node structure of MLP: " + str(nodestruct))

#########################

print("How many datasets do you want to use? (supply integer)")
trialstr = input()
trialnum = int(trialstr)

print("How do you want to use your own weights? (Y or N, if N then the weights will be randomly generated)")
ans = input()
######################## Opens and reads weights.

if ans == "Y":
    with open("file.txt", "r") as filew: # This code reads in the different float values in from a txt file.
        f_list = [float(i) for line in filew for i in line.split(' ') if i.strip()]
        mlpwarr = f_list[0:mlpweights]

    filew.close()
elif ans == "N":
    mlpwarr = np.random.rand(mlpweights)
else:
    print("Enter a correct ans input, N or Y")
    sys.exit(1) # apparently this works

######################### Opens and reads inputs.

with open("yinputs.txt", "r") as file1: # This code reads in the different float values in from a txt file.
    f_list = [float(i) for line in file1 for i in line.split(' ') if i.strip()]
    yinputs = f_list[0:int(nodestruct[0])*trialnum]

file1.close()

######################### Opens and reads the desired outputs.

with open("yactualout.txt", "r") as file2: # This code reads in the different float values in from a txt file.
    f_list = [float(i) for line in file2 for i in line.split(' ') if i.strip()]
    yactualout = f_list[0:int(nodestruct[-1])*trialnum]

file2.close()

######################### Definitions of some variables and arrays.


inputnum = int(nodestruct[0]) # number of inputs
outputs = int(nodestruct[layers-1]) # number of outputs in final layer.
yarr = np.zeros(elmntsinmlp) # Array that will contain all the information of the output of nodes, including the intial inputs.
deltal = np.zeros(elmntsinmlp) # Array that contains the deltal corresponding to a particular node, the array contains all deltal values in the node order.
totalloss = 0 # Variable used for calculating the lossfunction value.

######################### Passes all the inputs through the initial weights and prints out the initial values.

for trial in range(trialnum): # Cycles through the datasets.
    yarr = np.zeros(elmntsinmlp)

    for y in range(inputnum): # Cycles through the input values for a particular dataset.
        yinputpos = trial*inputnum + y # Calculates correct position of input in yinputs array.
        yarrpos = y # Position of the input in the yarr array.
        yarr[yarrpos] = yinputs[yinputpos]
    yarr = feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights) # output corresponding to particular input.
    print(yarr)
    print(trial)
    totalloss = 0
    for ifin in range(outputs): # loops through the final layers nodes.

        tlsyarrpos = elmntsinmlp-ifin-1
        totalloss += (yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])*(yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])

    print("totalloss before training: " + str(totalloss)) # Prints out the new loss.

######################### This the backpropagation training section of the code. 

for trial in range(trialnum): # Cycles through the datasets.
     
    yarr = np.zeros(elmntsinmlp)

    for y in range(inputnum): # Initializes the yarr
        yinputpos = trial*inputnum + y
        yarrpos = y
        yarr[yarrpos] = yinputs[yinputpos]
    
    yarr = feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights) # Passes yarr through untrained network.
    totalloss = 0

    for ifin in range(outputs): # Calculates the loss by looping through the final layer nodes.
        tlsyarrpos = elmntsinmlp-ifin-1 # Position of network output working backwards in comparision to forwardfeed node progression.
        yactpos = (trial+1)*outputs-ifin-1 # Corresponding position of the desired output.
        #print((trial+1)*outputs-ifin-1)
        totalloss += (yarr[tlsyarrpos] - yactualout[yactpos])*(yarr[tlsyarrpos] - yactualout[yactpos])

    dsdw = 0 # Change in the loss function for a change in weight
    lp = 1 # Learning parameter.

    ################ Firstlayer training:

    for ifin in range(outputs): # Cycles through the outputs.
        yacpos = (trial+1)*outputs-ifin-1
        yarrpos = elmntsinmlp-1-ifin

        yval = yarr[yarrpos]
        deltal[yarrpos] = (yval-yactualout[yacpos])*yval*(1-yval) # Assigns deltal to array in spot corresponding to the output node.

    weightpos = 0 # The position of the weight in question being altered.

    for lm1 in range(int(nodestruct[layers-2])): # Cycles through penultimate layer.
        for ifin in range(outputs): # Cycles through output layer
            dpos =  elmntsinmlp-1-ifin
            yarrpos = elmntsinmlp-outputs-1-lm1 # Cycles through node outputs of penultimate layer
            weightpos += -1

            dsdw = deltal[dpos]*yarr[yarrpos] 
            mlpwarr[weightpos] = mlpwarr[weightpos] - lp*dsdw # Adjusting the weight.

    print(yarr)
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
                mlpwpoint = weightstart2 -dl*int(nodestruct[layers-1-L])-1-di # I changed di*int(nodestruct[layers-2-L]) to dl*int(nodestruct[layers-1-L]) because the weight numbers that we are moving between are adjacent from how the 
                deltapoint = -dlstrt-di-1
                deltal[deltapoint2] += mlpwarr[mlpwpoint]*deltal[deltapoint]
                 

            deltal[deltapoint2] = deltal[deltapoint2]*yarr[deltapoint2]*(1-yarr[deltapoint2]) 

        #############################

        # Applies new dl values and calculates the dsdw values.
        for nodej in range(int(nodestruct[layers-3-L])): # For if you had multiple outputs.

            for dl in range(int(nodestruct[layers-2-L])):

                weightpoint = weightstart - dl - nodej*int(nodestruct[layers-2-L]) - 1
                yarrpoint = jstart - nodej - 1
                deltapoint = istart - dl - 1
                dsdw = deltal[deltapoint]*yarr[yarrpoint]
                mlpwarr[weightpoint] = mlpwarr[weightpoint] - (dsdw*lp)

        dlstrt += int(nodestruct[layers-1-L]) # alters the layer at which the dl values are calculated for.
        weightstart2 += -int(nodestruct[L-1]*nodestruct[L-2])
        istart += -int(nodestruct[layers-L-2])

    ########################### Calculating the new outputs and loss from the altered weights. 

    yarr = np.zeros(elmntsinmlp) # Re initializes yarr.
    for y in range(inputnum): # Inserts correct inputs corresponding to correct trial
        yinputpos = trial*inputnum + y # Calculates correct position of input in yinputs array.
        yarrpos = y
        yarr[yarrpos] = yinputs[yinputpos]

    yarr = feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights) # output corresponding to particular input.

    totalloss = 0
    for ifin in range(outputs): # loops through the final layers nodes.

        tlsyarrpos = elmntsinmlp-ifin-1
        totalloss += (yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])*(yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])

    print("totalloss after training: " + str(totalloss)) # Prints out the new loss.

############################ Passes all the data sets throught the final trained network.

for trial in range(trialnum): # Passes the inputs through the final trained network.
    yarr = np.zeros(elmntsinmlp)

    for y in range(inputnum): # Inserts correct inputs corresponding to correct trial
        yinputpos = trial*inputnum + y # Calculates correct position of input in yinputs array.
        yarrpos = y
        yarr[yarrpos] = yinputs[yinputpos]
    yarr = feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights) # output corresponding to particular input.
    

print("Final weights: ")
print(mlpwarr)

# Puts the new weights into a file called Nweights.txt with the same format as the original weights.txt file so that you can re use the new weights for more training or to save them if you have a finished product.
filenw = open("Nweights.txt","w")

for weight in range(mlpweights):
    filenw.write(str(mlpwarr[weight]) + " ")

filenw.close()
