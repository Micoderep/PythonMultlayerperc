from PIL import Image
import os

folder = "dp"

for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)

    if os.path.isfile(file_path):
        try:
            with Image.open(file_path) as img:
                if img.format not in ["JPEG", "JPG"]:
                    os.remove(file_path)
                    print(f"Deleted (not JPEG): {filename}")
        except Exception as e:
            # Corrupt or unsupported image
            os.remove(file_path)
            print(f"Deleted (could not open): {filename}")
