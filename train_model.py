import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

DATA_PATH = 'collected_data'

X = []
y = []

for gesture in os.listdir(DATA_PATH):
    for file in os.listdir(os.path.join(DATA_PATH, gesture)):
        data = np.load(os.path.join(DATA_PATH, gesture, file))
        X.append(data)
        y.append(gesture)

X = np.array(X)
y = np.array(y)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, 'gesture_model.pkl')

print("Model Trained Successfully!")