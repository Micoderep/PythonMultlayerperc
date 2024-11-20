Hi for this project I am going to code a multi-layer perceptron using Python for fun.

![image](https://github.com/user-attachments/assets/810ad0f7-bf43-40ea-93b9-08fa712e75ef)

(The image was found on this website: https://medium.com/codex/introduction-to-how-an-multilayer-perceptron-works-but-without-complicated-math-a423979897ac)

The movement forward is calculated as:

$$y = sig(\sum_{j=1}^{TNLj}(w_{ij}o_{j}))$$

Where $o_{j}$ is the output of a node from a layer and $w_{ij}$ is the weight corresponding to a connection of that node (j) with a node in the adjacent layer (i), the equation is summing the product of the output of all nodes in a prior layer with their corresponding weight to one node in the next layer. Say in the image above that we are trying to find the input of the top node in the central layer, we multiply the outputs of the left nodes with the weights which are represented by lines to the top central node. (TNLj is the total nodes in layer j)

Where "sig" is short for sigmoid an activation function that is defined as:

$$sig(o) = \frac{1}{1+e^{-o}}$$

![Logistic-curve](https://github.com/user-attachments/assets/b8a898fc-db02-456e-8a20-67abd3664682)

(The image was found on this website: https://en.wikipedia.org/wiki/Sigmoid_function)

The sigmoid function is used to provide a smooth differential function that quickly moves between the values 0 and 1 which mimicks the on or offness of neural networks. The differential aspect is necessary because for the back propagation algorithm to work the function must be differential. The activation function thus does not have to be a sigmoid function, other functions including ReLU are discussed here: https://www.v7labs.com/blog/neural-networks-activation-functions)

Back propagation is the name of the technique being used to train the neural network:
Loss function being used is the summed squared loss function, it bascially quantifies the difference between the desired output and the output calculated from passing a set of inputs through your current network.

$$ lossfunction = \sum_{l=1}^{TNFL}(o_{l}-ydesired_{l})^{2}$$

The back propagation optimises this in a way that always reduces the loss function via a method known as gradient descent. Gradient descent represented by this equation:

$$ wnew_{ij} = w_{ij} - lp*o_{j}dl_{i} $$

Where $wnew$ is the altered weight, $w$ is the original weight, $lp$ is called the learning parameter, $o_{j}$ is the output of the $j^{th}$ node in the $j^{th}$ layer, dl is just the symbol given to a collection of values put together as described by the equations below inorder that when multiplied by the correct output it adjusts the weight such that it reduces the loss function, bringing the network closer to producing the desired output for that set of inputs.

$$ dl_{l} = (o_{l} - ydesired_{l})o_{l}(1-o_{l}) $$

$$ dl_{i} = \sum_{l=1}^{TNLl}(w_{il}dl_{l})o_{i}(1-o_{i}) $$

In the two equations above the first is the dl calculated for altering the set of weights connecting the penultimate layer to the final layer, subsequent weights are altered using the second equation putting together the previous dl values created.

"Mlbp3.py" is the complete neural network, it takes in:

"mlpstruct.txt" file which contains the structure of the neural network in the format - a b c d.. where a b c d are integers of the number of nodes in a layer.

"yinputs.txt" file which contains all the inputs for all datasets pasted together.

"yactualout.txt" file which contains all the desired outputs for all datasets pasted together in the same file in the same order as the inputs.

"weights.txt" is an optional file which contains all of the weights of the neural network in an order such that in the image above weights are ordered going down in a particular layer then shifting to the top of the next layer. This is optional because I have made available an option of randomly generating the weights.

"Nweights.txt" is the file that the new trained weights go into.

Uses of neural network:

What I hope to achieve:

