import os
import numpy as np

def print_files_in_directory(directory):
    for filename in os.listdir(directory):
      arr = np.genfromtxt(d+"/"+filename, delimiter=',')
      print(arr.shape)


d = "blues_train_matrices"
print_files_in_directory(d)