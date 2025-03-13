#include <iostream>
#include <cmath>
#include <vector>
#include <fstream>
#include <sstream>
#include <random>
#include <memory>
#include <string>

double sigmoid(double x){ 
	return 1/(1+exp(-x));
}

void fileextract(std::string filename, std::unique_ptr<std::vector<double>>& array, int loops){
        
        double dvalue;
	std::ifstream file; file.open(filename);
        
	if ( file.is_open() ) { // Always check whether the file is open

		//while (file.good()){
		for(int num = 0; num <= loops; ++num){
			file >> dvalue;
			array->push_back(dvalue); // Since nodestruct's size not set it needs values appended to it using the push_back function otherwise there will be a segmentation error.
		}

                array->pop_back(); 
        }

	file.close();
}

class rng{
        // std::random variables (internal to class)
        std::mt19937 mt; // mersenne twister
        std::uniform_real_distribution<double> dist;
        public:
	        //seed rng with uniform distribution [0:1)
                void seed(unsigned int random_seed){
                        dist = std::uniform_real_distribution<double>(-1.0,1.0);
                        std::mt19937::result_type mt_seed = random_seed;
                        mt.seed(mt_seed); // seed generator
                        }// wrapper function generate a uniform random number between 0 and 1
                        double grnd(){
                                return dist(mt);
                        }
};

void feedforward(int layers,std::unique_ptr<std::vector<int>>& nodestruct, std::unique_ptr<std::vector<double>>& yarr, std::unique_ptr<std::vector<double>>& mlpwarr, int elmntsinmlp, int mlpweights){
	int presweight = 0; // Keeps track of which weight is to be used in forward passing.
	int yarrposi = 0; // Keeps track of position of new set of input nodes.
	int yarrposj = 0; // Keeps track of position of old set of input nodes.

	for(int L = 0; L < layers-1; ++L){ // Moves between layers denoted by indices 0 ... layer-2.
		int jrange = (*nodestruct)[L]; // int function as the values are floats otherwise and cannot be used for loop range.
		
		int irange = (*nodestruct)[L+1];
		yarrposi += jrange;        // Updates position to new set of nodes the information is flowing to.
			
			for(int j = 0; j < jrange; ++j){ // Cycles through the inputs from prior layer. 
				int yjpos = yarrposj + j;
				
				for(int i = 0; i < irange; ++i){ // Cycles through the nodes of present layer, will be using to cycle through weights.
					int yipos = yarrposi + i;
					(*yarr)[yipos] += ((*mlpwarr)[presweight])*((*yarr)[yjpos]);
					
					presweight += 1; // Keeps incrementing the weight index.
				}

                        }
			
			for(int i = 0; i < irange; ++i){ // Applies sigmoid function to the sum of the inputs for new output.
				int yipos = yarrposi + i;
				(*yarr)[yipos] = sigmoid((*yarr)[yipos]);
			}

			yarrposj += jrange;        // Updates positon of old nodes.

        }
}


int main(int argc, char **argv){
	
        std::unique_ptr<std::vector<int>> nodestruct = std::make_unique<std::vector<int>>();	
        std::unique_ptr<std::vector<double>> mlpwarr = std::make_unique<std::vector<double>>();	
	//std::vector<int> nodestruct; // Creates a vector to contain the number of nodes per layer.
	//std::vector<double> mlpwarr;
        int value, layers, elmntsinmlp = 0, mlpweights, trialnum; // value is a variable for a stream value to be put into before being allocated to a vector.
        double dvalue;
        rng random; // Makes an rng object.

	std::string yon;

	std::cout << "How many datasets from the available data do you want to train the neural network on?" << std::endl;
	std::cin >> trialnum;

	std::cout << "Do you want to use random weights or pre-existing ones? (R or P)" << std::endl;

	std::cin >> yon;

///////////////////////////////////////////////////////////////////////////////////

	std::ifstream mlpstruct; mlpstruct.open("mlpstruct.txt"); // Creates a stream and opens the mlpstruct.txt file.
	if ( mlpstruct.is_open() ) { // Always check whether the file is open
		while (mlpstruct.good()){
			mlpstruct >> value;
			nodestruct->push_back(value); // Since nodestruct's size not set it needs values appended to it using the push_back function otherwise there will be a segmentation error.
		}
	}
        mlpstruct.close(); 

        nodestruct->pop_back(); // Corrects the repetitive value at the end of the vector.
	layers = nodestruct->size(); // Number of layers in the neural network.
	
	std::cout << "layers: " << layers << std::endl;
///////////////////////////////////////////////////////////////////////////////////
        mlpweights = 0;

        for(int num = 0; num < layers; ++num){ // Calculates the total number of nodes in the neural network.
		elmntsinmlp = elmntsinmlp + (*nodestruct)[num];
                if (num != 0){
			mlpweights += ((*nodestruct)[num])*((*nodestruct)[num-1]);
		}
        } 

/////////////////////////////////////////////////////////////////////////////////// 

	if (yon == "R"){
		random.seed(1);

		for(int num = 0; num < mlpweights; ++num){
			mlpwarr->push_back(random.grnd());
	        }

	} else if (yon == "P"){

        fileextract("fil3.txt", mlpwarr, mlpweights);

	} else{
		std::cout << "Program did not receive an acceptable input (R or P)" << std::endl;
		exit(0);
	}

///////////////////////////////////////////////////////////////////////////////////

	std::unique_ptr<std::vector<double>> yarr = std::make_unique<std::vector<double>>(elmntsinmlp);
        std::unique_ptr<std::vector<double>> yinputs = std::make_unique<std::vector<double>>();
        std::unique_ptr<std::vector<double>> yactualout = std::make_unique<std::vector<double>>();   

///////////////////////////////////////////////////////////////////////////////////

        fileextract("yinputs.txt", yinputs, ((*nodestruct)[0])*trialnum); // yinput and yactualout pointer being allocated the file data.

        fileextract("yactualout.txt", yactualout, ((*nodestruct)[layers-1])*trialnum);
        
	if (yinputs->size() == ((*nodestruct)[0])*trialnum && yactualout->size() == ((*nodestruct)[layers-1])*trialnum){

	} else{
		std::cout << "inputs or outputs size does not match with number of trials or nodestructure." << std::endl;
		exit(0);

	}

//////////////////////////////////////////////////////////////////////////////////

	int inputnum = ((*nodestruct)[0]), outputs = ((*nodestruct)[layers-1]), yinputpos, yarrpos, tlsyarrpos, yacpos, yactpos, dpos, weightpos;
	double totalloss, lp = 1, dsdw, yval;
        std::vector<double> deltal(elmntsinmlp); // Array that contains the deltal corresponding to a particular node, the array contains all deltal values in the node order.
        int weightstart, weightstart2, istart, jstart, dlstrt, deltapoint, deltapoint2, yarrpoint, weightpoint, mlpwpoint;
////////////////////////// Passes all the inputs through the initial weights and prints out the initial values.

	for(int trial = 0; trial < trialnum; ++trial){ // Cycles through the datasets.

		for(int num = 0; num < elmntsinmlp; ++num){
			(*yarr)[num] = 0.0;
		}
                
		for(int y = 0; y < inputnum; ++y){ // Cycles through the input values for a particular dataset.
			yinputpos = trial*inputnum + y; // Calculates correct position of input in yinputs array.
			yarrpos = y; // Position of the input in the yarr array.
			(*yarr)[yarrpos] = (*yinputs)[yinputpos];
		}

		feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights); // output corresponding to particular input.
		
		totalloss = 0;
		for(int ifin = 0; ifin < outputs; ++ifin){ // loops through the final layers nodes.
			
			tlsyarrpos = elmntsinmlp-ifin-1;
			totalloss += ((*yarr)[tlsyarrpos] - (*yactualout)[(trial+1)*outputs-ifin-1])*((*yarr)[tlsyarrpos] - (*yactualout)[(trial+1)*outputs-ifin-1]);
	                //std::cout << "totalloss: " << totalloss << std::endl;
	        }
        }


//////////////////////////////////////////////////// Begin training.
        for(int trial = 0; trial < trialnum; ++trial){ // Cycles through the datasets.
     
		for(int num = 0; num < elmntsinmlp; ++num){
			(*yarr)[num] = 0.0;
			deltal[num] = 0.0;
		}
		
		for(int y = 0; y < inputnum; ++y){ // Initializes the yarr
			yinputpos = trial*inputnum + y;
			yarrpos = y;
			(*yarr)[yarrpos] = (*yinputs)[yinputpos];
                } 
		
		feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights); // Passes yarr through untrained network.
		totalloss = 0;
                
		for(int ifin = 0; ifin < outputs; ++ifin){ // Calculates the loss by looping through the final layer nodes.
			tlsyarrpos = elmntsinmlp-ifin-1; // Position of network output working backwards in comparision to forwardfeed node progression.
			yactpos = (trial+1)*outputs-ifin-1; // Corresponding position of the desired output.
			totalloss += ((*yarr)[tlsyarrpos] - (*yactualout)[yactpos])*((*yarr)[tlsyarrpos] - (*yactualout)[yactpos]);
                }
		
		dsdw = 0; // Change in the loss function for a change in weight

//////////////////////////////////////////////////////////////	Firstlayer training:
                
		for(int ifin = 0; ifin < outputs; ++ifin){ // Cycles through the outputs.
			    yacpos = (trial+1)*outputs-ifin-1;
                yarrpos = elmntsinmlp-1-ifin;
			
			    yval = (*yarr)[yarrpos];
			    deltal[yarrpos] = (yval-(*yactualout)[yacpos])*yval*(1-yval); // Assigns deltal to array in spot corresponding to the output node.
			    //std::cout << yval << " " << yactualout[yacpos] << " " << deltal[yarrpos] << std::endl;
        }
		
		weightpos = mlpweights; // The position of the weight in question being altered.
		
		for(int lm1 = 0; lm1 < (*nodestruct)[layers-2]; ++lm1){ // Cycles through penultimate layer.
			for(int ifin = 0; ifin < outputs; ++ifin){ // Cycles through output layer.
				dpos = elmntsinmlp-1-ifin;
				yarrpos = elmntsinmlp-outputs-1-lm1; // Cycles through node outputs of penultimate layer
				
				weightpos += -1;
				
				dsdw = deltal[dpos]*(*yarr)[yarrpos];
				(*mlpwarr)[weightpos] = (*mlpwarr)[weightpos] - lp*dsdw; // Adjusting the weight.
                        }
		}

//////////////////////////////////////////////////////////// For all other layers:
        istart = -(*nodestruct)[layers-1];
	dlstrt = elmntsinmlp;
	jstart = -(*nodestruct)[layers-1];
	weightstart = mlpweights;
	weightstart2 = mlpweights;

	for(int L = 0; L <= layers-2; ++L){ //-2 as the first iteration has already been done above and there are no weights before the first layer.
	    weightstart += -((*nodestruct)[layers-L-1])*((*nodestruct)[layers-L-2]);
		jstart += -(*nodestruct)[layers-L-2];

        //////////////////////////////////////////////////////
        // Calculates the new dl values from layer change
	        for(int dl = 0; dl < (*nodestruct)[layers-2-L]; ++dl){ // Adds weighted dl's in previous layer to make new ones
                // Cycles through different dl values adds the ones from different nodes in the previous layer.
		        deltapoint2 = elmntsinmlp-1+istart-dl;
		        for(int di = 0; di < (*nodestruct)[layers-1-L]; ++di){ // Cycles through nodes in previous layer
		        mlpwpoint = weightstart2 -dl*((*nodestruct)[layers-1-L])-1-di; // I changed di*int(nodestruct[layers-2-L]) to dl*int(nodestruct[layers-1-L]) because the weight numbers that we are moving between are adjacent from how the 
		        deltapoint = dlstrt-di-1;
				deltal[deltapoint2] += ((*mlpwarr)[mlpwpoint])*deltal[deltapoint];
				
//				if (deltapoint2 < 5){
//					std::cout << "mlpwarr: " << mlpwarr[mlpwpoint] << " mlpwpoint: " << mlpwpoint << " deltal: " << deltal[deltapoint] << " deltapoint: " << deltapoint << " di: " << di << " dl: " << dl << " deltapoint2: " << deltapoint2 << std::endl;
//				}
				//std::cout << "dlstrt: " << dlstrt << " dpoint: " << deltapoint << " dpoint2: " << deltapoint2 << " mlpwpoint: " << mlpwpoint << " mlpwarr: " << mlpwarr[mlpwpoint] << " dlval: " << deltal[deltapoint] << " ws: " << weightstart << " ws2: " << weightstart2 << " trial: " << trial << " L: " << L <<std::endl;
                        }                 
                        
			deltal[deltapoint2] = deltal[deltapoint2]*((*yarr)[deltapoint2])*(1-(*yarr)[deltapoint2]);
		}	
			//	for(int num = 0; num < elmntsinmlp; ++num){
			//	    std::cout << deltal[num] << " " << trial << std::endl;
            // 	}

        //////////////////////////////////////////////////////

        // Applies new dl values and calculates the dsdw values.
	        for(int nodej = 0; nodej < (*nodestruct)[layers-3-L]; ++nodej){ // For if you had multiple outputs.

                        for(int dl = 0; dl < (*nodestruct)[layers-2-L]; ++dl){
                            weightpoint = weightstart - dl - nodej*((*nodestruct)[layers-2-L]) - 1;
                            yarrpoint = elmntsinmlp+jstart - nodej - 1;
                            deltapoint = elmntsinmlp+istart - dl - 1;
                            dsdw = deltal[deltapoint]*((*yarr)[yarrpoint]);
                            (*mlpwarr)[weightpoint] = (*mlpwarr)[weightpoint] - (dsdw*lp);
                            
                                                    
                            //std::cout << "deltal: " << deltal[deltapoint] << " deltapoint: " << deltapoint << " dl: " << dl << " trial: " << trial << std::endl;			    
                            //std::cout << "dlstrt: " << dlstrt << " dpoint: " << deltapoint << " deltal: " << deltal[deltapoint] << " dpoint2: " << deltapoint2 << " mlpwpoint: " << mlpwpoint << " mlpwarr: " << mlpwarr[weightpoint] << " dsdw: " << dsdw << " ws: " << weightstart << " ws2: " << weightstart2 << " yarrpoint: " << yarrpoint << " yarr: " << yarr[yarrpoint] <<" trial: " << trial << std::endl;
                            //std::cout << "dlstrt: " << dlstrt << " dpoint: " << deltapoint << " dpoint2: " << deltapoint2 << " mlpwpoint: " << mlpwpoint << " ws: " << weightstart << " ws2: " << weightstart2 << " yarrpoint: " << yarrpoint << " trial: " << trial << std::endl;
                        }
		}
		dlstrt += -(*nodestruct)[layers-1-L]; // alters the layer at which the dl values are calculated for.
		weightstart2 += -((*nodestruct)[layers-L-1])*((*nodestruct)[layers-L-2]);
		istart += -(*nodestruct)[layers-L-2];

        } /////////////////////////////////////////// End of training.

        
	for(int num = 0; num < elmntsinmlp; ++num){
		(*yarr)[num] = 0.0;
	}
    
	for(int y = 0; y < inputnum; ++y){ // Inserts correct inputs corresponding to correct trial
		yinputpos = trial*inputnum + y; // Calculates correct position of input in yinputs array.
                yarrpos = y;
                (*yarr)[yarrpos] = (*yinputs)[yinputpos];
        }
        feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights); // output corresponding to particular input.

        totalloss = 0;
        for(int ifin = 0; ifin < outputs; ++ifin){ // loops through the final layers nodes.
		tlsyarrpos = elmntsinmlp-ifin-1;
		totalloss += ((*yarr)[tlsyarrpos] - (*yactualout)[(trial+1)*outputs-ifin-1])*((*yarr)[tlsyarrpos] - (*yactualout)[(trial+1)*outputs-ifin-1]);

	}
        }
        
        for(int num = 0; num < mlpweights; ++num){
		std::cout << "mlpwarr: " << (*mlpwarr)[num] << std::endl;
	}

	std::ofstream Nweights("Nweights.txt");

        for(int num = 0; num < elmntsinmlp; ++num){
	        Nweights << (*mlpwarr)[num] << " ";	
	}
	std::cout << deltal[0] << " " << deltal[1] << std::endl;

        Nweights.close();
        
    for(int num = 0; num < elmntsinmlp; ++num){
		std::cout << "deltal: " << deltal[num] << std::endl;
	}
    for(int num = 0; num < elmntsinmlp; ++num){
		std::cout << "yarr: " << (*yarr)[num] << std::endl;
	}
	return 0;
}


