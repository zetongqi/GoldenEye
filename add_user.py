import pickle
import os
import sys
import shutil
import PIL
import numpy as np
from mtcnn.mtcnn import MTCNN
from numpy import linalg as LA


DIR = os.path.dirname(os.path.abspath(__file__))
USER_DATABASE = os.path.join(DIR, "user_database")
model_path = os.path.join(DIR, "facenet_keras.h5")
NUM_BASE = 3

def get_face_array(img_path):
    image = PIL.Image.open(img_path)
    image = image.convert('RGB')
    pixels = np.asarray(image)
    detector = MTCNN()
    results = detector.detect_faces(pixels)
    x1, y1, width, height = results[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = pixels[y1:y2, x1:x2]
    image = PIL.Image.fromarray(face)
    image = image.resize((160, 160))
    face_array = np.asarray(image)
    mean, std = face_array.mean(), face_array.std()
    face_array = (face_array - mean) / std
    return np.expand_dims(face_array, axis=0).astype("float32")

if __name__ == "__main__":
    import tensorflow as tf
    model = tf.keras.models.load_model(model_path)
    for layer in model.layers:
        layer.trainable=False

    with open('users.pickle', 'rb') as handle:
        users = pickle.load(handle)

    with open('d.pickle', 'rb') as handle:
        d = pickle.load(handle)

    face_ary_sum = 0
    new_id = str(len(users.keys()))
    for base_id, file_loc in zip(range(1, NUM_BASE+1), sys.argv[1:-1]):
    	path = os.path.join(USER_DATABASE, "img_database"+str(base_id))
    	filename = os.path.join(path, str(new_id)+".jpg")
    	print(filename)
    	shutil.copy(file_loc, filename)
    	print("file: " + file_loc + " added to the database" + str(base_id))
    	face_ary_sum += model.predict(get_face_array(filename))

    d[new_id] = face_ary_sum / 3
    users[new_id] = sys.argv[-1]

    # re-pickle
    with open('d.pickle', 'wb') as handle:
        pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    with open('users.pickle', 'wb') as handle:
        pickle.dump(users, handle, protocol=pickle.HIGHEST_PROTOCOL)

