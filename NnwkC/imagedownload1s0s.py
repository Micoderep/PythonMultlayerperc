import os
from keras.datasets import mnist
from PIL import Image

# Load MNIST data
(train_images, train_labels), _ = mnist.load_data()

# Create output directories
os.makedirs("mnist_ones", exist_ok=True)
os.makedirs("mnist_zeros", exist_ok=True)

# Save '1's and '0's
ones_saved = 0
zeros_saved = 0

for i, (img, label) in enumerate(zip(train_images, train_labels)):
    if label == 1 and ones_saved < 1000:
        Image.fromarray(img).save(f"mnist_ones/one_{ones_saved:04d}.jpg")
        ones_saved += 1
    elif label == 0 and zeros_saved < 1000:
        Image.fromarray(img).save(f"mnist_zeros/zero_{zeros_saved:04d}.jpg")
        zeros_saved += 1
    if ones_saved == 1000 and zeros_saved == 1000:
        break

print("Done! Saved 1000 images of 1's and 0's.")
