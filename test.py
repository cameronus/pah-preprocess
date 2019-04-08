import numpy as np

data = np.load('datasets/training_n1000-m10-p2_2019-04-02_03-14PM.npz')
print(len(data['x']))
print(len(data['y']))
# print(data['y'][0])
print(len(data['x'][0]))
