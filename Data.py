import os
import numpy as np
import torch


blues_label = np.ones(len(os.listdir('blues_train_vectors')))
rock_label = np.zeros(len(os.listdir('rock_train_vectors')))

y_vec = np.concatenate((blues_label, rock_label))
print(y_vec)

Y = torch.tensor(y_vec)

data = []


def add_data(dir_path, data_list):
  for filename in os.listdir(dir_path):
    arr = np.genfromtxt(dir_path+"/"+filename, delimiter=',')
    data_list.append(arr)

add_data('blues_train_vectors', data)
add_data('rock_train_vectors', data)

np_arr = np.array(data)
X = torch.tensor(np_arr).float()