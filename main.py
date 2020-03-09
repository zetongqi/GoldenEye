import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
import PIL
import os
import pickle
from numpy import linalg as LA

dir_path = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(dir_path, "facenet_keras.h5")
model = tf.keras.models.load_model(model_path)
for layer in model.layers:
    layer.trainable=False
model.summary()

with open('users.pickle', 'rb') as handle:
    users = pickle.load(handle)


cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
detector = MTCNN()

with open('d.pickle', 'rb') as handle:
    d = pickle.load(handle)

def process_frame(frame):
    frame1 = np.array([frame[:, :, -1], frame[:, :, 1], frame[:, :, 0]])
    frame1 = np.transpose(frame1, (1, 2, 0))
    return frame1

def get_face_embedding(face_ary, model):
    return model.predict(face_ary)

def search(dic, q):
    vals = {}
    for key in dic.keys():
        vals[key] = LA.norm(dic[key] - q)
    if vals[min(vals, key=vals.get)] >= 10:
        return "NOT_REG"
    else:
        return min(vals, key=vals.get)

while(True):
    ret, frame = cap.read()
    frame1 = process_frame(frame)
    results = detector.detect_faces(frame1)

    for i in range(len(results)):
        x1, y1, width, height = results[i]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
        face = frame1[y1:y2, x1:x2]
        image = PIL.Image.fromarray(face)
        image = image.resize((160, 160))
        face_array = np.asarray(image)
        mean, std = face_array.mean(), face_array.std()
        face_array = (face_array - mean) / std
        q = model.predict(np.expand_dims(face_array, axis=0))
        res = search(d, q)
        cv2.putText(frame, users[res], (x1+5,y1-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (153,255,51), 2)  


    cv2.imshow('video', frame)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()