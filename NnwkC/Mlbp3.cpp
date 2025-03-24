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

void fileextract(std::string filename, std::unique_ptr<std::vector<double>>& array){ // int loops
        
        double dvalue;
	std::ifstream file; file.open(filename);
        
	if ( file.is_open() ) { // Always check whether the file is open

		while (file.good()){
		//for(int num = 0; num <= loops; ++num){
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

class AI{

	private:


	public:
                
		// 1 pass of the data through the network for a particular set of the data + the totalloss of this pass is calcd. 
		void pass1(std::unique_ptr<std::vector<double>>& yarr, std::unique_ptr<std::vector<double>>& mlpwarr, std::unique_ptr<std::vector<double>>& yinputs, std::unique_ptr<std::vector<double>>& yactualout, int trial, int inputnum, int outputs, int mlpweights, std::unique_ptr<double>& totalloss, int layers, std::unique_ptr<std::vector<int>>& nodestruct, int elmntsinmlp){ 
			int yinputpos, yarrpos, tlsyarrpos;
			
			for(int num = 0; num < elmntsinmlp; ++num){
				(*yarr)[num] = 0.0;
			}
			
			for(int y = 0; y < inputnum; ++y){ // Cycles through the input values for a particular dataset.
				yinputpos = trial*inputnum + y; // Calculates correct position of input in yinputs array.
				yarrpos = y; // Position of the input in the yarr array.
				(*yarr)[yarrpos] = (*yinputs)[yinputpos];
			}
			
			feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights); // output corresponding to particular input.
			
			*totalloss = 0;
			for(int ifin = 0; ifin < outputs; ++ifin){ // loops through the final layers nodes.
				tlsyarrpos = elmntsinmlp-ifin-1;
				*totalloss += ((*yarr)[tlsyarrpos] - (*yactualout)[(trial+1)*outputs-ifin-1])*((*yarr)[tlsyarrpos] - (*yactualout)[(trial+1)*outputs-ifin-1]);
			}
		}
		
		// Trains the neural network on all the specified datasets.
		void training(std::unique_ptr<std::vector<double>>& yarr, std::unique_ptr<std::vector<double>>& mlpwarr, std::unique_ptr<std::vector<double>>& yinputs, std::unique_ptr<std::vector<double>>& yactualout, int trialnum1, int trialnum2, int inputnum, int outputs, int mlpweights, std::unique_ptr<double>& totalloss, int layers, std::unique_ptr<std::vector<int>>& nodestruct, int elmntsinmlp){
			
			double dsdw, yval, lp = 1.0;
			int yacpos, yarrpos, dpos, weightpos, istart, dlstrt, jstart, weightstart, weightstart2, deltapoint, deltapoint2, mlpwpoint, weightpoint, yarrpoint, trial;
			std::vector<double> deltal(elmntsinmlp);
			
// Start of the training.
//
//
			for(int trial = trialnum1; trial < trialnum2; ++trial){ // Cycles through the datasets.
				
				pass1(yarr, mlpwarr, yinputs, yactualout, trial, inputnum, outputs, mlpweights, totalloss, layers, nodestruct, elmntsinmlp);
				
				for(int num = 0; num < elmntsinmlp; ++num){
					deltal[num] = 0.0;
				}
				
				dsdw = 0; // Change in the loss function for a change in weight

// Firstlayer training.
//
//
                                for(int ifin = 0; ifin < outputs; ++ifin){ // Cycles through the outputs.
					yacpos = (trial+1)*outputs-ifin-1;
					yarrpos = elmntsinmlp-1-ifin;
					yval = (*yarr)[yarrpos];
					deltal[yarrpos] = (yval-(*yactualout)[yacpos])*yval*(1-yval); // Assigns deltal to array in spot corresponding to the output node.
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

// For all other layers.
//
//
                                
				istart = -(*nodestruct)[layers-1];
				dlstrt = elmntsinmlp;
				jstart = -(*nodestruct)[layers-1];
				weightstart = mlpweights;
				weightstart2 = mlpweights;
				
				for(int L = 0; L <= layers-2; ++L){ //-2 as the first iteration has already been done above and there are no weights before the first layer.
					weightstart += -((*nodestruct)[layers-L-1])*((*nodestruct)[layers-L-2]);
					jstart += -(*nodestruct)[layers-L-2];
// Calculates the new dl values from layer change.
	                                for(int dl = 0; dl < (*nodestruct)[layers-2-L]; ++dl){ 
// Cycles through different dl values adds the ones from different nodes in the previous layer.
		                        
						deltapoint2 = elmntsinmlp-1+istart-dl;
						for(int di = 0; di < (*nodestruct)[layers-1-L]; ++di){ // Cycles through nodes in previous layer
							mlpwpoint = weightstart2 -dl*((*nodestruct)[layers-1-L])-1-di;
							deltapoint = dlstrt-di-1;
							deltal[deltapoint2] += ((*mlpwarr)[mlpwpoint])*deltal[deltapoint];
						}                 
						
						deltal[deltapoint2] = deltal[deltapoint2]*((*yarr)[deltapoint2])*(1-(*yarr)[deltapoint2]);
					}	

// Applies new dl values and calculates the dsdw values.
	                                
					for(int nodej = 0; nodej < (*nodestruct)[layers-3-L]; ++nodej){ // Cycles through all nodes corresponding the to layer for which the weight improvements are being applied to.
						for(int dl = 0; dl < (*nodestruct)[layers-2-L]; ++dl){
							weightpoint = weightstart - dl - nodej*((*nodestruct)[layers-2-L]) - 1;
							yarrpoint = elmntsinmlp+jstart - nodej - 1;
							deltapoint = elmntsinmlp+istart - dl - 1;
							dsdw = deltal[deltapoint]*((*yarr)[yarrpoint]);
							(*mlpwarr)[weightpoint] = (*mlpwarr)[weightpoint] - (dsdw*lp);
						}
					}
					
					dlstrt += -(*nodestruct)[layers-1-L]; // alters the layer at which the dl values are calculated for.
					weightstart2 += -((*nodestruct)[layers-L-1])*((*nodestruct)[layers-L-2]);
					istart += -(*nodestruct)[layers-L-2];
				}
			}
		}
};

int main(int argc, char **argv){
	
        std::unique_ptr<std::vector<int>> nodestruct = std::make_unique<std::vector<int>>();	
        std::unique_ptr<std::vector<double>> mlpwarr = std::make_unique<std::vector<double>>();	
        int value, layers, elmntsinmlp = 0, mlpweights; // value is a variable for a stream value to be put into before being allocated to a vector.
        double fraction;
	int epochnum;
	rng random; // Makes an rng object.
        AI ai;

	std::cout << "How many data training epochs do you want?" << std::endl;
	std::cin >> epochnum;
	std::cout << "What fraction of the data do you want to train the data on? (The other will be testing)" << std::endl;
	std::cin >> fraction;

	std::cout << "Do you want to use random weights or pre-existing ones? (R or P)" << std::endl;

	std::string yon;
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
	
// Calculating the total number of nodes in the neural network elmntsinmlp, and the total number of weights, mlpweights.
//
//
        mlpweights = 0;

        for(int num = 0; num < layers; ++num){ // Calculates the total number of nodes in the neural network.
		elmntsinmlp = elmntsinmlp + (*nodestruct)[num];
                if (num != 0){
			mlpweights += ((*nodestruct)[num])*((*nodestruct)[num-1]);
		}
        } 

// Code that either calculates random weights or extracts them from a file depending on the user input.
//
//

	if (yon == "R"){
		random.seed(1);

		for(int num = 0; num < mlpweights; ++num){
			mlpwarr->push_back(random.grnd());
	        }

	} else if (yon == "P"){

        fileextract("fil3.txt", mlpwarr);

	} else{
		std::cout << "Program did not receive an acceptable input (R or P)" << std::endl;
		exit(0);
	}

// Defines smart pointer vectors which contains all the inputs, outputs, and contemporary node output values.
//
//

	std::unique_ptr<std::vector<double>> yarr = std::make_unique<std::vector<double>>(elmntsinmlp);
        std::unique_ptr<std::vector<double>> yinputs = std::make_unique<std::vector<double>>();
        std::unique_ptr<std::vector<double>> yactualout = std::make_unique<std::vector<double>>();   

// Extracts all the inputs and outputs and checks if the number matches up with the nodestruct information.
//
//

        fileextract("yinputs.txt", yinputs); //  ((*nodestruct)[0])*trialnum
        fileextract("yactualout.txt", yactualout); //  ((*nodestruct)[layers-1])*trialnum
        
	if ((yinputs->size())%((*nodestruct)[0]) == 0 && (yactualout->size())%((*nodestruct)[layers-1]) == 0){

	} else if((yinputs->size())%((*nodestruct)[0]) != 0){
		std::cout << "inputs extracted are not a factor of the number of first layer nodes." << std::endl;
                exit(0);
	       
	} else{
		std::cout << "inputs or outputs size does not match with number of trials or nodestructure." << std::endl;
		exit(0);

	}
        
//////////////////////////////////////////////////////////////////////////////////

	int inputnum = ((*nodestruct)[0]), outputs = ((*nodestruct)[layers-1]);
	std::unique_ptr<double> totalloss = std::make_unique<double>();

//Passes all the inputs through the initial weights and prints out the initial values.
//
//

	for(int trial = 0; trial < 4; ++trial){ // Cycles through the datasets.		
		ai.pass1(yarr, mlpwarr, yinputs, yactualout, trial, inputnum, outputs, mlpweights, totalloss, layers, nodestruct, elmntsinmlp);        
        }


// Training.
//
//
	int TTritpepoch = round((fraction*(yinputs->size()))/epochnum), TTitpepoch = (yinputs->size())-TTritpepoch*epochnum;

        for(int epoch = 0; epoch < epochnum; ++epoch){
		std::cout << "epoch: " << epoch << std::endl;
		int trialnum1 = TTritpepoch*epoch, trialnum2 = TTritpepoch*(epoch+1); 
		ai.training(yarr, mlpwarr, yinputs, yactualout, trialnum1, trialnum2, inputnum, outputs, mlpweights, totalloss, layers, nodestruct, elmntsinmlp);

		for(int trial = TTritpepoch*epochnum; trial < yinputs->size(); ++trial){
			std::cout << "Trial: " << trial << std::endl;
			ai.pass1(yarr, mlpwarr, yinputs, yactualout, trial, inputnum, outputs, mlpweights, totalloss, layers, nodestruct, elmntsinmlp);
		}

	}
// Calculating out the last iteration of yarr for checking.
//
//
        for(int trial = TTritpepoch*epochnum-1; trial < TTritpepoch*epochnum; ++trial){ // Passes the final training set through the trained NNwk.
		ai.pass1(yarr, mlpwarr, yinputs, yactualout, trial, inputnum, outputs, mlpweights, totalloss, layers, nodestruct, elmntsinmlp);
	}
// Printing all the final weights and yarr.
//
//
        for(int num = 0; num < mlpweights; ++num){
		std::cout << "mlpwarr: " << (*mlpwarr)[num] << std::endl;
	}
        for(int num = 0; num < elmntsinmlp; ++num){
		std::cout << "yarr: " << (*yarr)[num] << std::endl;
	}

// Writing the new weights to a file for potential later use.
//
//
	std::ofstream Nweights("Nweights.txt");
        for(int num = 0; num < elmntsinmlp; ++num){
	        Nweights << (*mlpwarr)[num] << " ";	
	}
        Nweights.close();

	return 0;
}


