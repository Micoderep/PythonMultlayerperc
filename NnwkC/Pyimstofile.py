import random
import numpy as np
from PIL import Image
import os
import sys

folder_path = "mnist_ones"
filesf = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
fplen = len(filesf)

folder_path = "mnist_zeros"
filesd = sorted([f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))])
dplen = len(filesd)

totalimages = fplen + dplen

print("totalimages: " + str(totalimages) + " fplen: " + str(fplen) + " dplen: " + str(dplen))

print("fract1: " + str(dplen/totalimages) + " fract2: " + str(fplen/totalimages))
fpcount = 0
dpcount = 0

with open("data.txt", "a") as totaldata:
    with open("dataout.txt", "a") as dataout:

        for i in range(totalimages):
            ran = random.random()

            if ran < fplen/totalimages and ran >= 0 and fpcount < fplen: # The flower directory is opened.
                image_path = "mnist_ones/"+ str(filesf[fpcount]) # Replace with your image file
                img = Image.open(image_path)
                resized_image = img.resize((28, 28))
                # Convert to grayscale (optional, if you only need brightness values
                grim = resized_image.convert("L")
                #arrayshape = pixels.shape
                grim = np.array(grim)
                data = grim.flatten()
                data = data/255 # To normalize all the images so that their pixels have relative size and so that they dont overwhelm the weak weights of the neural network.
#                print(data)
                print("ran: " + str(ran) + " fpcount: " + str(fpcount) + " fraction: " + str(fplen/totalimages) + " i: " + str(i)) 
                
                if np.amax(data) == 1 and np.amin(data) == 1:
                    continue

                np.savetxt(totaldata, [data], fmt='%.6f', newline=' ')
                dataout.write(str(1) + " ")
                
                fpcount += 1

            elif ran > fplen/totalimages and ran <= 1.0 and dpcount < dplen: # The daffodil image directory is opened.
                image_path = "mnist_zeros/" + str(filesd[dpcount])  # Replace with your image file
                img = Image.open(image_path)
                resized_image = img.resize((28, 28))
                # Convert to grayscale (optional, if you only need brightness values
                grim = resized_image.convert("L")
                #arrayshape = pixels.shape
                grim = np.array(grim)
                data = grim.flatten()
                data = data/255

                print("ran: " + str(ran) + " dpcount: " + str(dpcount) + " fraction: " + str(dplen/totalimages) + " i: " + str(i)) 
                if np.amax(data) == 1 and np.amin(data) == 1:
                    continue
                np.savetxt(totaldata, [data], fmt='%.6f', newline=' ')
                dataout.write(str(0) + " ")
                dpcount += 1
            elif fpcount == fplen or dpcount == dplen:
                break
            else:
                print("Something went wrong, maybe ran's value is not acceptable: " + str(ran))
                sys.exit()

