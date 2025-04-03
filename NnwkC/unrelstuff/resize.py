from PIL import Image

# Load an image
image = Image.open('daffs1.jpeg')

# Resize to 224x224
resized_image = image.resize((100, 100))

# Save or show
resized_image.save('daffs1rs.jpeg')
resized_image.show()
