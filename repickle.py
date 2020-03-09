import numpy as np
import tensorflow as tf
from mtcnn.mtcnn import MTCNN
import PIL
import os
from numpy import linalg as LA
import pickle

from add_user import get_face_array

dir_path = os.path.dirname(os.path.realpath(__file__))
database_path = os.path.join(dir_path, "user_database", "img_database")
model_path = os.path.join(dir_path, "facenet_keras.h5")
NUM_DATABASE = 3

model = tf.keras.models.load_model(model_path)
for layer in model.layers:
    layer.trainable=False
    

d = {}
for i in range(1, NUM_DATABASE+1):
    path = database_path + str(i) + "/"
    for file in os.listdir(path):
        if "jpg" in file:
            idx = file.replace(".jpg", "")
            if idx not in d.keys():
                d[idx] = model.predict(get_face_array(path+file))
            else:
                d[idx] += model.predict(get_face_array(path+file))

for key in d.keys():
    d[key] = d[key] / NUM_DATABASE
    
print("pickling user data")
with open('d.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
