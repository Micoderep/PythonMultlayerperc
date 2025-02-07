from PIL import Image
import numpy as np

# Load image
image_path = "daffs1.jpeg"  # Replace with your image file
img = Image.open(image_path)

# Convert to grayscale (optional, if you only need brightness values)
img = img.convert("L")

# Convert image to a NumPy array
pixels = np.array(img)

arrayshape = pixels.shape

maxsize = 2048


pixels = np.pad(pixels, [(0, maxsize-arrayshape[0]), (0, maxsize-arrayshape[1])], mode='constant')
print(arrayshape[1])
# Flatten the pixel values into a single column
#pixel_column = pixels.flatten()
pixel_row = pixels.flatten()

# Save pixel values to a TXT file
output_file = "pixel_data.txt"
#np.savetxt(output_file, pixel_column, fmt="%d")
np.savetxt(output_file, [pixel_row], fmt="%d", delimiter=" ")
print(f"Pixel data saved to {output_file}")

