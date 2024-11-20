![Logistic-curve](https://github.com/user-attachments/assets/b8a898fc-db02-456e-8a20-67abd3664682)Hi for this project I am going to code a multi-layer perceptron using Python for fun.

![image](https://github.com/user-attachments/assets/810ad0f7-bf43-40ea-93b9-08fa712e75ef)
(The image was found on this website: https://medium.com/codex/introduction-to-how-an-multilayer-perceptron-works-but-without-complicated-math-a423979897ac)

The movement forward is calculated as:

$$y = sig(\sum_{j=1}^{TNLj}(w_{ij}x_{j}))$$

Where $x_{j}$ is the output of a node from a layer and $w_{ij}$ is the weight corresponding to a connection of that node (j) with a node in the adjacent layer (i), the equation is summing the product of the output of all nodes in a prior layer with their corresponding weight to one node in the next layer. Say in the image above that we are trying to find the input of the top node in the central layer, we multiply the outputs of the left nodes with the weights which are represented by lines to the top central node.

Where "sig" is short for sigmoid an activation function that is defined as:

$$sig(x) = \frac{1}{1+e^{-x}}$$

![Uploading Logistic-cu<?xml version="1.0" encoding="utf-8"  standalone="no"?>
<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" 
    "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd">
<svg width="600" height="400" viewBox="0 0 600 400"
    xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink"  >
  <title>Graph of Logistics Curve</title>
  <desc>Originally Produced by GNUPLOT 4.2 patchlevel 2, hand compressed  </desc>

  <g style="fill:none; stroke-width:1.00; stroke-linecap:butt">
    <g stroke="#888">
      <path d='M38.5,346.8 H 561.5 ' />
      <path d='M38.5,185.5 H 561.5 ' />
      <path d='M38.5, 24.2 H 561.5 ' />
      <path d='M 38.5,346.0 V 25.0 ' />
      <path d='M125.7,346.0 V 25.0 ' />
      <path d='M212.8,346.0 V 25.0 ' />
      <path d='M300.0,346.0 V 25.0 ' />
      <path d='M387.2,346.0 V 25.0 ' />
      <path d='M474.3,346.0 V 25.0 ' />
      <path d='M561.5,346.0 V 25.0 ' />
    </g>
    <g stroke="#000" stroke-width="2" >
      <path d='M300.0,346.8 h-12 M300.0,185.5 h-12 M300.0, 24.2 h-12' />
      <path d='M 38.5,346.0 v-14 M125.7,346.0 v-14 M212.8,346.0 v-14
          M300.0,346.0 v-14 M387.2,346.0 v-14 M474.3,346.0 v-14 M561.5,346.0 v-14' />
      <path d='M300.0,346.0 V 25.0 M38.5, 346.0 H 561.5' />
    </g>
  </g>
  <g style="stroke:none; fill:black; font-family:Deja Vu Sans,Lucida Sans; font-size:18.00pt">
    <g text-anchor="end" transform="translate(287,0)" >
      <text y="353.5" >0</text>
      <text y="192.2">0.5</text>
      <text y="30.9">1</text>
    </g>
    <g text-anchor="middle" transform="translate(0,379.7)" >
      <text x="38.5" >−6</text>
      <text x="125.7" >−4</text>
      <text x="212.8" >−2</text>
      <text x="300.0" > 0</text>
      <text x="387.2" > 2</text>
      <text x="474.3" > 4</text>
      <text x="561.5" > 6</text>
    </g>
  </g>
  <g fill="none" stroke-width="2.0"  transform="translate(168.6,249.5)scale(.995,1.038)" >
    <path stroke="#4050C0" d="M-132.0,93.0 L-107.8,92.4 L-94.6,92.0 L-81.4,91.3 L-68.2,90.5
      L-55.0,89.4 L-41.8,87.9 L-28.6,85.9 L-15.4,83.2 L-2.2,79.7 L11.0,75.0 L24.2,69.0
      C28.7,66.7 33.1,64.1 37.4,61.3 C41.9,58.3 46.3,55.0 50.6,51.5
      C55.1,47.7 59.5,43.6 63.8,39.3 C68.3,34.6 72.7,29.7 77.0,24.4
      C81.5,18.9 85.9,13.0 90.2,6.9
      L103.4,-13.0 L116.6,-34.9 L129.8,-58.0 L138.6,-73.5
      L151.8,-96.3 L165.0,-117.6 L178.2,-136.8
      C182.5,-142.6 186.9,-148.2 191.4,-153.4 C195.7,-158.4 200.1,-163.0 204.6,-167.4
      C208.9,-171.5 213.3,-175.3 217.8,-178.7 C222.1,-182.1 226.5,-185.1 231.0,-187.8
      C235.3,-190.5 239.8,-192.8 244.2,-195.0
      L257.4,-200.5 L270.6,-204.7 L283.8,-207.9 L297.0,-210.3 L310.2,-212.2 L323.4,-213.5
      L336.6,-214.6 L349.8,-215.3 L363.0,-215.9 L374.0,-216.2 L393.8,-216.7" />
  </g>
</svg>

rve.svg…]()


Back propagation is the name of the technique being used to train the neural network:
Loss function being used is the summed squared loss function, it bascially quantifies the difference between the desired output and the output calculated from passing a set of inputs through your current network.

$$ lossfunction = \sum_{j=1}^{TNFL}(youtput-ydesired)^{2} $$

The back propagation optimises this in a way that always reduces the loss function via a method known as gradient descent.

//work in progress

$$ dl = (youtput - ydesired)youtput(1-youtput) $$

$$ dl = \sum_{l=1}^{TNLl}(w_{il}dl_{l})youtput(1-youtput) $$

"Mlbp3.py" is the complete neural network, it takes in:

"mlpstruct.txt" file which contains the structure of the neural network in the format - a b c d.. where a b c d are integers of the number of nodes in a layer.

"yinputs.txt" file which contains all the inputs for all datasets pasted together.

"yactualout.txt" file which contains all the desired outputs for all datasets pasted together in the same file in the same order as the inputs.

"weights.txt" is an optional file which contains all of the weights of the neural network in an order such that in the image above weights are ordered going down in a particular layer then shifting to the top of the next layer. This is optional because I have made available an option of randomly generating the weights.

"Nweights.txt" is the file that the new trained weights go into.

Uses of neural network:

What I hope to achieve:

