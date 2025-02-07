import numpy as np
from PIL import Image

# Load pixel values from file
pixel_data = np.loadtxt("pixel_data.txt", dtype=np.uint8)

# Define the original image shape (YOU MUST KNOW THIS)
image_width = 384  # Replace with actual width
image_height = 512  # Replace with actual height
image_width = 2048  # Replace with actual width
image_height = 2048  # Replace with actual height

# Reshape the pixel data to match the original image
image_array = pixel_data.reshape((image_height, image_width))

# Convert array back to image
reconstructed_image = Image.fromarray(image_array)

# Save or show the image
reconstructed_image.save("reconstructed_image.jpeg")
reconstructed_image.show()

