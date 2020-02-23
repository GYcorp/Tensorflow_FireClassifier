import cv2
import numpy as np

dataset = np.load('./dataset/train_pos.npy', allow_pickle = True)

for data in dataset:
    image_path, flame, smoke = data
    print(data)
    input()
