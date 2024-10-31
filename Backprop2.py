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
deltal = np.zeros(int(elmntsinmlp*trialnum)) # Array that contains the deltal corresponding to a particular node in a particular layer.

totalloss, yarr = trialruns(trialnum,inputnum,yarr,yinputs,elmntsinmlp,outputs,layers,nodestruct,mlpwarr,mlpweights)
#print(yarr)
#print("Now weights")
#print(mlpwarr)

#print("Final value of input: " + str(totalloss))


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
            deltal[yinodepos] = (yval-yactualout[yacpos])*yval*(1-yval) # Assigns deltal to array in spot corresponding to the output node.
            #print((yval-yactualout[yacpos])*yval*(1-yval)*yarr[yjnodepos])
            #print(trialnum)
            #print("trial: " + str(trial))
            #print("mlpwbefore: " + str(mlpwarr[presweight]))
            mlpwarr[presweight] = mlpwarr[presweight] - (lp*dsdw)
            #print("mlpwafter: " + str(mlpwarr[presweight]) + " Weight: " + str(presweight))

            presweight = -1*(nodej+1)-(trial+1)*int(mlpweights)
            dsdw = 0


#print(deltal)
# More layers:
istart = -int(nodestruct[layers-1])
dlstrt = 0
jstart = -int(nodestruct[layers-1])
weightstart = 0

for L in range(layers-2): #-1 as the first iteration has already been done above.
    weightstart += -int(nodestruct[L-1]*nodestruct[L-2])

    jstart += -int(nodestruct[layers-L-2])
    ###############################
    # Calculates the new dl values from layer change
    for trial in range(trialnum): # loops over all trials for 
        #yinodepos = int(elmntsinmlp*(trialnum-trial)+istart-1) # Starts from the end of each set and works up
        #yjnodepos = int(elmntsinmlp*(trialnum-trial)-nodej+jstart-1) # position of the correct prior layer node corresponding to the weight currently being changed.
        #yval = yarr[yinodepos]

        #print("Trial: " + str(trial))    
        #print(deltal)
        for dl in range(int(nodestruct[layers-2-L])): # Adds weighted dl's in previous layer to make new ones
            # Cycles through different dl values adds the ones from different nodes in the previous layer.
                
            deltapoint2 = elmntsinmlp*(trialnum-trial)-1+istart-dl
            for di in range(int(nodestruct[layers-1-L])): # Cycles through nodes in previous layer
                mlpwpoint = -int(mlpweights)*trial-dl-1-di*int(nodestruct[layers-2-L])
                deltapoint = elmntsinmlp*(trialnum-trial)-dlstrt-di-1

                deltal[deltapoint2] += mlpwarr[mlpwpoint]*deltal[deltapoint]
                 
                #print("trial: "+ str(trial) + " dlstrt: " + str(dlstrt))
                #print("trial: "+ str(trial) + " dl: " + str(dl) + " di: " + str(di) +" mlpwpoint: "+ str(mlpwpoint) + " deltapoint: " + str(deltapoint) + " deltal2: " + str(deltal[deltapoint]) + " deltal2: " + str(deltal[deltapoint2]))
                #print("trial: "+ str(trial) + " dl: " + str(dl) + " di: " + str(di) +" mlpwpoint: "+ str(mlpwpoint) +" deltapoint: " + str(deltapoint) + " deltapoint2: " + str(deltapoint2))
        print("Trial: " + str(trial))    
        print(deltal)
        for dl in range(int(nodestruct[layers-2-L])):
            deltapoint = elmntsinmlp*(trialnum - trial) - dl - 1 + istart

            print("trial: " + str(trial) + " dl: " + str(dl) + " deltapoint: " + str(deltapoint) + " yarr: " + str(yarr[deltapoint]) + " deltal: " + str(deltal[deltapoint]))
            deltal[deltapoint] = deltal[deltapoint]*yarr[deltapoint]*(1-yarr[deltapoint]) 

            print("trial: " + str(trial) + " dl: " + str(dl) + " deltapoint: " + str(deltapoint) + " yarr: " + str(yarr[deltapoint]) + " deltal: " + str(deltal[deltapoint]))
    #############################
    # Applies new dl values and calculates the dsdw values.
        for nodej in range(int(nodestruct[layers-3-L])): # For if you had multiple outputs.
            #istart += -int(nodestruct[layers-L-1])
            #for nodei in range(int(nodestruct[layers-2-L])): # Loop for second to last layer!
            #presweight = -int(weightstart)-nodej*int(nodestruct[layers-2-L])-1 # Starts at weight -4 as last weight was -3, the product calculates the number of weights between the last and penultimate layer.
            #yval = yarr[yinodepos]

            for dl in range(int(nodestruct[layers-2-L])): # Adds weighted dl's in previous layer to make new ones
            # Cycles through different dl values adds the ones from different nodes in the previous layer.
                
                #deltapoint2 = elmntsinmlp*(trialnum-trial)-1+istart-dl
                #deltal[] = deltal[]*yval*(1-yval)
                #dsdw = deltal[deltapoint2]*yarr[yjnodepos]
                #mlpwarr[presweight] = mlpwarr[presweight] - (lp*dsdw)
                #print(presweight)
                #presweight += -int(mlpweights)

                weightpoint = weightstart - dl - int(mlpweights)*(trial) - nodej*int(nodestruct[layers-2-L]) - 1
                yarrpoint = jstart - nodej - elmntsinmlp*(trial) - 1
                deltapoint = istart - dl - elmntsinmlp*(trial) - 1
                #print("istart: "+str(istart) + " jstart: " + str(jstart) +" yarr: " + str(yarrpoint) +" deltapoint: " + str(deltapoint)+ " weightpoint: " + str(weightpoint))
                #print("mlpwarr before: " + str(mlpwarr[weightpoint]) + " weightpoint: " + str(weightpoint))
                #print("deltapoint: " + str(deltapoint) + " yarrpoint: " + str(yarrpoint) + " yarr: " + str(yarr[yarrpoint]) + " deltal: " + str(deltal[deltapoint]))
                dsdw = deltal[deltapoint]*yarr[yarrpoint]
                mlpwarr[weightpoint] = mlpwarr[weightpoint] - (dsdw*lp)
                #print("mlpwarr: " + str(mlpwarr[weightpoint]) + " dsdw: " + str(dsdw))
    dlstrt += int(nodestruct[layers-1-L]) # alters the layer at which the dl values are calculated for.

weightsum = 0
print("Now weights")
print(mlpwarr)
for w in range(int(mlpweights)):

    for trial in range(trialnum):
        weightpoint = w + int(mlpweights)*trial 
        if weightpoint != w:
            mlpwarr[w] += mlpwarr[weightpoint]

print("mid weights")
print(mlpwarr)
for w in range(int(mlpweights)):

    for trial in range(trialnum):
        weightpoint = w + trial*int(mlpweights)
        mlpwarr[weightpoint] = mlpwarr[w]

mlpwarr = mlpwarr/trialnum
#mlpwarr = (mlpwarr)/trialnum

totalloss, yarr = trialruns(trialnum,inputnum,yarr,yinputs,elmntsinmlp,outputs,layers,nodestruct,mlpwarr,mlpweights)

print(deltal)
print(yarr)
print("Now weights")
print(mlpwarr)
print("Final value of input: " + str(totalloss))
