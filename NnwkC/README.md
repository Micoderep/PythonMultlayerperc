Here is the description of the above code and how to use it:

First the files known as data and dataout.txt are created by running Pyimstofile.py. What this code does is takes the images in mnist_ones and mnist_zeros and pastes them randomly into a row (The randomness is weighted to the dataset which has more data to allow complete spread of the different data when training) in the data.txt file subsequent images are appended on to the end. For each image appended, a 0 or a 1 corresponding to an image of a 0 or a 1 is appended to the row in the dataout.txt file.

Next compile and run the Mlbp3.cpp file which trains the neural network on the data in data.txt and the actual outputs in dataout.txt. The training are done in periods called epochs where a portion of the data is trained and the resulting neural network is tested on the whole fraction of the data not allocated to training. Losses are computed for each epoch and are printed to a file called losses.txt and are plotted using the plot.gnu file (The graph is displayed below), in addition the resulting weights are printed in a file called Nweights.txt.

![Lpfnt](https://github.com/user-attachments/assets/bb01b243-1706-4880-898f-7550d3a9ee94)

This graph shows how the testing and training losses vary as the code progresses for a network of structure: 748, 128, 64, 1 (these are found in the mlpstructd.txt file), you can see that the losses decrease but then increase a bit towards the end.

In this folder there are images; synthetic_one, zero, and generated_zero (shown below), using jpegtotxt.py these can be turned into row txt files that can be processed by the resultant weights of the training by compiling and running the feed.cpp file. This file outputs the output of the neural network for the particular image input to the terminal.

![synthetic_one](https://github.com/user-attachments/assets/a4a7efa7-4304-45c7-8b5d-d5fdadfa4414)
![synthetic_zero](https://github.com/user-attachments/assets/6fe8ee39-e9c2-4ceb-a30c-0058a179d791)
![zero_generated](https://github.com/user-attachments/assets/3182a1b7-c419-41b2-8e5e-1f2698ddaebb)

The middle input image has an unexpectedly high value ~0.7-0.8, I tested by using an input image (right most one) with a zero with a similar style to the training data and it gave a value much closer to zero ~0.001.

