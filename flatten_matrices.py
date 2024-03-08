import os
import numpy as np

def flatten_matrices(matrix_directory, vector_directory):
    for filename in os.listdir(matrix_directory):
      arr = np.genfromtxt(matrix_directory+"/"+filename, delimiter=',')
      vec = arr.ravel()
      np.savetxt(vector_directory+"/"+filename, vec, delimiter=",")



d = "blues_train_matrices"
v_d = "blues_train_vectors"

rock_matrix_dir = 'rock_train_matrices'
rock_vec = 'rock_train_vectors'
# flatten_matrices(d, v_d)
flatten_matrices(rock_matrix_dir, rock_vec)