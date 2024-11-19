Hi for this project I am going to code a multi-layer perceptron using Python for fun.

![image](https://github.com/user-attachments/assets/810ad0f7-bf43-40ea-93b9-08fa712e75ef)
(The image was found on this website: https://medium.com/codex/introduction-to-how-an-multilayer-perceptron-works-but-without-complicated-math-a423979897ac)

The movement forward is calculated as:

$$y = sig(\sum_{j=0}^{totalnodeslayerj}(w_{ij}x_{j}))$$

Where $x_{j}$ is the output of a node from a layer and $w_{ij}$ is the weight corresponding to a connection of that node (j) with a node in the adjacent layer (i), the equation is summing the product of the output of all nodes in a prior layer with their corresponding weight to one node in the next layer. Say in the image above that we are trying to find the input of the top node in the central layer, we multiply the outputs of the left nodes with the weights which are represented by lines to the top central node.

Where "sig" is short for sigmoid an activation function that is defined as:

$$sig(x) = \frac{1}{1+e^{-x}}$$

Back propagation is the name of the technique being used to train the neural network:
Loss function being used is the summed squared loss function, it bascially quantifies the difference between the desired output and the output calculated from passing a set of inputs through your current network.

$$ lossfunction = \sum_{j=0}^{totalnodesinfinallayer}(youtput-ydesired)^{2} $$

The back propagation optimises this in a way that always reduces the loss function via a method known as gradient descent.

$$ dl = $$


