from PIL import Image
import numpy as np
from FilePaths import blues_paths, rock_paths

def write_data_for_category(image_paths, prefix, new_dir):
  for image_path in image_paths:
    image_big = Image.open(image_path)
    new_width = 40
    new_height = 60
    image = image_big.resize((new_width, new_height), Image.LANCZOS)
    # Convert the image to grayscale if it's not already (optional, depending on your needs)
    image_gray = image.convert('L')
    image_array = np.array(image_gray)
    img_name = (image_path.removeprefix(prefix)).removesuffix(".jpg")
    np.savetxt(new_dir+'/'+img_name+".csv", image_array, delimiter=",")

write_data_for_category(blues_paths, 'train/Blues/', 'blues_train_matrices')
write_data_for_category(rock_paths, 'train/Rock/', 'rock_train_matrices')
