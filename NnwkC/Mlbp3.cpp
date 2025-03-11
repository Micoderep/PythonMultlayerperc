#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>

double sigmoid(double x){ 
	return 1/(1+exp(-x));
}

std::vector<double> feedforward(int layers,std::vector<int> nodestruct, std::vector<double> yarr, std::vector<double> mlpwarr, int elmntsinmlp, int mlpweights){
	int presweight = 0; // Keeps track of which weight is to be used in forward passing.
	int yarrposi = 0; // Keeps track of position of new set of input nodes.
	int yarrposj = 0; // Keeps track of position of old set of input nodes.

	for(int L = 0; L < layers-1; ++L){ // Moves between layers denoted by indices 0 ... layer-2.
		int jrange = nodestruct[L]; // int function as the values are floats otherwise and cannot be used for loop range.
		
		int irange = nodestruct[L+1];
		yarrposi += jrange;        // Updates position to new set of nodes the information is flowing to.
			
			for(int j = 0; j < jrange; ++j){ // Cycles through the inputs from prior layer. 
				int yjpos = yarrposj + j;
				
				for(int i = 0; i < irange; ++i){ // Cycles through the nodes of present layer, will be using to cycle through weights.
					int yipos = yarrposi + i;
					yarr[yipos] += mlpwarr[presweight]*yarr[yjpos];
					
					presweight += 1; // Keeps incrementing the weight index.
				}

                        }
			
			for(int i = 0; i < irange; ++i){ // Applies sigmoid function to the sum of the inputs for new output.
				int yipos = yarrposi + i;
				yarr[yipos] = sigmoid(yarr[yipos]);
			}

			yarrposj += jrange;        // Updates positon of old nodes.

        }
	return yarr;
}


int main(int argc, char **argv){
	
	std::ifstream mlpstruct; mlpstruct.open("mlpstruct.txt"); // Creates a stream and opens the mlpstruct.txt file.
	std::vector<int> nodestruct; // Creates a vector to contain the number of nodes per layer.
	std::vector<double> mlpwarr;
        int value, layers, elmntsinmlp = 0, mlpweights, trialnum; // value is a variable for a stream value to be put into before being allocated to a vector.
        double dvalue;

	std::cout << "How many datasets from the available data do you want to train the neural network on?" << std::endl;
	std::cin >> trialnum;

///////////////////////////////////////////////////////////////////////////////////
	
	if ( mlpstruct.is_open() ) { // Always check whether the file is open
		while (mlpstruct.good()){
			mlpstruct >> value;
			nodestruct.push_back(value); // Since nodestruct's size not set it needs values appended to it using the push_back function otherwise there will be a segmentation error.
		}
	}
        mlpstruct.close(); 

        nodestruct.pop_back(); // Corrects the repetitive value at the end of the vector.
	layers = nodestruct.size(); // Number of layers in the neural network.

///////////////////////////////////////////////////////////////////////////////////

        for(int num = 0; num < layers; ++num){ // Calculates the total number of nodes in the neural network.
		elmntsinmlp = elmntsinmlp + nodestruct[num]; 
        } 

///////////////////////////////////////////////////////////////////////////////////

        std::ifstream weights; weights.open("weights.txt");

	if ( weights.is_open() ) { // Always check whether the file is open
		while (weights.good()){
			weights >> dvalue;
			mlpwarr.push_back(dvalue); // Since nodestruct's size not set it needs values appended to it using the push_back function otherwise there will be a segmentation error.
		}
	}

	weights.close();
        mlpwarr.pop_back(); 
	mlpweights = mlpwarr.size();

///////////////////////////////////////////////////////////////////////////////////

	std::vector<double> yarr(elmntsinmlp), yinputs(nodestruct[0]*trialnum), yactualout(nodestruct[layers-1]*trialnum); // Because we know the total number of nodes in the neural network we can now initialize the yarr vector which will contain all the inputs/outputs provided to and calculated by the network. 

///////////////////////////////////////////////////////////////////////////////////

	std::ifstream yinputsf; yinputsf.open("yinputs.txt");

	if ( yinputsf.is_open() ) { // Always check whether the file is open

		for(int i = 0; i < nodestruct[0]*trialnum; ++i){
			yinputsf >> dvalue;
			yinputs[i] = dvalue; // Since nodestruct's size not set it needs values appended to it using the push_back function otherwise there will be a segmentation error.
			//std::cout << yinputs[i] << std::endl;
		}

        }

	yinputsf.close();

////////////////////////////////////////////////////////////////////////////////////

        for(int num = 0; num < elmntsinmlp; ++num){
		yarr[num] = 0.0;
	}

	std::ifstream yactualoutf; yactualoutf.open("yactualout.txt");
        
	if ( yactualoutf.is_open() ) { // Always check whether the file is open

		for(int i = 0; i < nodestruct[layers-1]*trialnum; ++i){
			yactualoutf >> dvalue;
			yactualout[i] = dvalue; // Since nodestruct's size not set it needs values appended to it using the push_back function otherwise there will be a segmentation error.
			//std::cout << yactualout[i] << std::endl;
		}

        }

	yactualoutf.close();

/////////////////////////////////////////////////////////////////////////////////////
        
	int inputnum = nodestruct[0], outputs = nodestruct[layers-1], yinputpos, yarrpos, tlsyarrpos;
	double totalloss;
        std::vector<double> deltal(elmntsinmlp); // Array that contains the deltal corresponding to a particular node, the array contains all deltal values in the node order.

////////////////////////// Passes all the inputs through the initial weights and prints out the initial values.

	for(int trial = 0; trial < trialnum; ++trial){ // Cycles through the datasets.

		for(int num = 0; num < elmntsinmlp; ++num){
			yarr[num] = 0.0;
		}
                
		for(int y = 0; y < inputnum; ++y){ // Cycles through the input values for a particular dataset.
			yinputpos = trial*inputnum + y; // Calculates correct position of input in yinputs array.
			yarrpos = y; // Position of the input in the yarr array.
			yarr[yarrpos] = yinputs[yinputpos];
		}

		yarr = feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights); // output corresponding to particular input.
                
		for(int num = 0; num < elmntsinmlp; ++num){
			std::cout << trial << " " << num << " " << yarr[num] << std::endl;
		}


		totalloss = 0;
		for(int ifin = 0; ifin < outputs; ++ifin){ // loops through the final layers nodes.
			
			tlsyarrpos = elmntsinmlp-ifin-1;
			totalloss += (yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])*(yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1]);
	                std::cout << "totalloss: " << totalloss << std::endl;
	        }
        }

	//for(int i = 0; i < elmntsinmlp; ++i){
	//	std::cout << "yarr:" << yarr[i] << std::endl;
	//}
	//for(int i = 0; i < outputs*trialnum; ++i){
	//	std::cout << "yactualout:" << yactualout[i] << std::endl;
	//}

//////////////////////////////////////////////////// Begin training.

        for(int trial = 0; trial < trialnum; ++trial){ // Cycles through the datasets.
     
		for(int num = 0; num < elmntsinmlp; ++num){
			yarr[num] = 0.0;
		}
		
		for(int y = 0; y < inputnum; ++y){ // Initializes the yarr
			int yinputpos = trial*inputnum + y;
			int yarrpos = y;
			yarr[yarrpos] = yinputs[yinputpos];
                } 
		
		yarr = feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights); // Passes yarr through untrained network.
		totalloss = 0;
                
		for(int ifin = 0; ifin < outputs; ++ifin){ // Calculates the loss by looping through the final layer nodes.
			int tlsyarrpos = elmntsinmlp-ifin-1; // Position of network output working backwards in comparision to forwardfeed node progression.
			int yactpos = (trial+1)*outputs-ifin-1; // Corresponding position of the desired output.
			totalloss += (yarr[tlsyarrpos] - yactualout[yactpos])*(yarr[tlsyarrpos] - yactualout[yactpos]);
                }
		
		double dsdw = 0; // Change in the loss function for a change in weight
		double lp = 1; // Learning parameter.

//////////////////////////////////////////////////////////////	Firstlayer training:
                
		for(int ifin = 0; ifin < outputs; ++ifin){ // Cycles through the outputs.
			int yacpos = (trial+1)*outputs-ifin-1;
                        int yarrpos = elmntsinmlp-1-ifin;
			
			double yval = yarr[yarrpos];
			deltal[yarrpos] = (yval-yactualout[yacpos])*yval*(1-yval); // Assigns deltal to array in spot corresponding to the output node.
                }
		
		int weightpos = mlpweights; // The position of the weight in question being altered.
		
		for(int lm1 = 0; lm1 < nodestruct[layers-2]; ++lm1){ // Cycles through penultimate layer.
			for(int ifin = 0; ifin < outputs; ++ifin){ // Cycles through output layer.
				int dpos = elmntsinmlp-1-ifin;
				yarrpos = elmntsinmlp-outputs-1-lm1; // Cycles through node outputs of penultimate layer
				
				weightpos += -1;
				
				dsdw = deltal[dpos]*yarr[yarrpos];
				mlpwarr[weightpos] = mlpwarr[weightpos] - lp*dsdw; // Adjusting the weight.
                        }
		}
                
		for(int i = 0; i < elmntsinmlp; ++i){
			std::cout << yarr[i] << std::endl;
	        }	

//////////////////////////////////////////////////////////// For all other layers:
        int istart = -int(nodestruct[layers-1]);
	int dlstrt = 0;
	int jstart = -int(nodestruct[layers-1]);
	int weightstart = 0;
	int weightstart2 = 0;

	for(int L = 0; L < layers-2; ++L){ //-2 as the first iteration has already been done above and there are no weights before the first layer.
	        weightstart += -int(nodestruct[L-1]*nodestruct[L-2]);
		jstart += -nodestruct[layers-L-2];

        //////////////////////////////////////////////////////

        // Calculates the new dl values from layer change
	        for(int dl = 0; dl < nodestruct[layers-2-L]; ++dl){ // Adds weighted dl's in previous layer to make new ones
                // Cycles through different dl values adds the ones from different nodes in the previous layer.
		        int deltapoint2 = -1+istart-dl;
		        for(int di = 0; di < nodestruct[layers-1-L]; ++di){ // Cycles through nodes in previous layer
		                int mlpwpoint = weightstart2 -dl*int(nodestruct[layers-1-L])-1-di; // I changed di*int(nodestruct[layers-2-L]) to dl*int(nodestruct[layers-1-L]) because the weight numbers that we are moving between are adjacent from how the 
				int deltapoint = -dlstrt-di-1;
				deltal[deltapoint2] += mlpwarr[mlpwpoint]*deltal[deltapoint];
                        }                 
                        
			deltal[deltapoint2] = deltal[deltapoint2]*yarr[deltapoint2]*(1-yarr[deltapoint2]);
		}	

        //////////////////////////////////////////////////////

        // Applies new dl values and calculates the dsdw values.
	        for(int nodej = 0; nodej < int(nodestruct[layers-3-L]); ++nodej){ // For if you had multiple outputs.

                        for(int dl = 0; dl < int(nodestruct[layers-2-L]); ++dl){
			        
			        int weightpoint = weightstart - dl - nodej*int(nodestruct[layers-2-L]) - 1;
				int yarrpoint = jstart - nodej - 1;
				int deltapoint = istart - dl - 1;
				int dsdw = deltal[deltapoint]*yarr[yarrpoint];
				mlpwarr[weightpoint] = mlpwarr[weightpoint] - (dsdw*lp);
                        }
		}
		dlstrt += int(nodestruct[layers-1-L]); // alters the layer at which the dl values are calculated for.
		weightstart2 += -int(nodestruct[L-1]*nodestruct[L-2]);
		istart += -int(nodestruct[layers-L-2]);

        } /////////////////////////////////////////// End of training.

        
	for(int num = 0; num < elmntsinmlp; ++num){
		yarr[num] = 0.0;
	}
    
	for(int y = 0; y < inputnum; ++y){ // Inserts correct inputs corresponding to correct trial
		yinputpos = trial*inputnum + y; // Calculates correct position of input in yinputs array.
                yarrpos = y;
                yarr[yarrpos] = yinputs[yinputpos];
        }
        yarr = feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights); // output corresponding to particular input.

        totalloss = 0;
        for(int ifin = 0; ifin < outputs; ++ifin){ // loops through the final layers nodes.
		tlsyarrpos = elmntsinmlp-ifin-1;
		totalloss += (yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1])*(yarr[tlsyarrpos] - yactualout[(trial+1)*outputs-ifin-1]);

	        std::cout << "totalloss after training: " << totalloss << std::endl;
	}
        }
	return 0;
}
