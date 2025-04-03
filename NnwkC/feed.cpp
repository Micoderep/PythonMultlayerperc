
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
          	                        //std::cout << "presweight: " << presweight << " yipos: " << yipos << " i: " <<  i << std::endl;
				}

                        }
			
			for(int i = 0; i < irange; ++i){ // Applies sigmoid function to the sum of the inputs for new output.
				int yipos = yarrposi + i;
				(*yarr)[yipos] = sigmoid((*yarr)[yipos]);
			}

			yarrposj += jrange;        // Updates positon of old nodes.

        }
}

int main(){
	
	std::unique_ptr<std::vector<int>> nodestruct = std::make_unique<std::vector<int>>();	
        std::unique_ptr<std::vector<double>> mlpwarr = std::make_unique<std::vector<double>>();	
        int value, layers, elmntsinmlp = 0, mlpweights; // value is a variable for a stream value to be put into before being allocated to a vector.
	
	std::ifstream mlpstruct; mlpstruct.open("mlpstructd.txt"); // Creates a stream and opens the mlpstruct.txt file.
	if ( mlpstruct.is_open() ) { // Always check whether the file is open
		while (mlpstruct.good()){
			mlpstruct >> value;
			nodestruct->push_back(value); // Since nodestruct's size not set it needs values appended to it using the push_back function otherwise there will be a segmentation error.
		}
	}
        mlpstruct.close(); 

        nodestruct->pop_back(); // Corrects the repetitive value at the end of the vector.
	layers = nodestruct->size(); // Number of layers in the neural network.
	
        mlpweights = 0;

        for(int num = 0; num < layers; ++num){ // Calculates the total number of nodes in the neural network.
		elmntsinmlp = elmntsinmlp + (*nodestruct)[num];
                if (num != 0){
			mlpweights += ((*nodestruct)[num])*((*nodestruct)[num-1]);
		}
        }

        fileextract("Nweights.txt", mlpwarr);

	std::cout << "mlpweights: " << mlpweights << " elmntsinmlp: " << elmntsinmlp << std::endl;
	std::cout << mlpwarr->size() << std::endl;

	std::unique_ptr<std::vector<double>> yarr = std::make_unique<std::vector<double>>(elmntsinmlp);
        std::unique_ptr<std::vector<double>> yinputs = std::make_unique<std::vector<double>>();
        std::unique_ptr<std::vector<double>> yactualout = std::make_unique<std::vector<double>>();   

// Extracts all the inputs and outputs and checks if the number matches up with the nodestruct information.
//
//

        fileextract("0.txt", yinputs); //  ((*nodestruct)[0])*trialnum
//        fileextract("dataout.txt", yactualout); //  ((*nodestruct)[layers-1])*trialnum


	int inputnum = (*nodestruct)[0], yarrpos, yinputpos;
	for(int num = 0; num < elmntsinmlp; ++num){
		(*yarr)[num] = 0.0;
	}
			
	for(int y = 0; y < inputnum; ++y){ // Cycles through the input values for a particular dataset.
                yinputpos = y; // Calculates correct position of input in yinputs array.
		yarrpos = y; // Position of the input in the yarr array.
		(*yarr)[yarrpos] = (*yinputs)[yinputpos];
	}
			
	feedforward(layers,nodestruct,yarr,mlpwarr,elmntsinmlp,mlpweights); // output corresponding to particular input.

	std::cout << (*yarr)[elmntsinmlp-1] <<std::endl;
			
	return 0;
}
