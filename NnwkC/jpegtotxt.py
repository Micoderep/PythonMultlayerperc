from PIL import Image
import numpy as np

# Load image
image_path = "synthetic_zero.jpg"  # Replace with your image file
img = Image.open(image_path)

# Convert to grayscale (optional, if you only need brightness values)
img = img.convert("L")

# Convert image to a NumPy array
pixels = np.array(img)

pixels = pixels/255

#arrayshape = pixels.shape

#maxsize = 2048


#pixels = np.pad(pixels, [(0, maxsize-arrayshape[0]), (0, maxsize-arrayshape[1])], mode='constant')
#print(arrayshape[1])
# Flatten the pixel values into a single column
#pixel_column = pixels.flatten()
pixel_row = pixels.flatten()

# Save pixel values to a TXT file
output_file = "0.txt"
#np.savetxt(output_file, pixel_column, fmt="%d")
np.savetxt(output_file, [pixel_row], fmt="%.6f", delimiter=" ")
print(f"Pixel data saved to {output_file}")

