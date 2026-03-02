import cv2
import os
import mediapipe as mp
import numpy as np

gesture_name = input("Enter Gesture Name: ")

DATA_PATH = os.path.join('collected_data', gesture_name)
os.makedirs(DATA_PATH, exist_ok=True)

print("Opening camera...")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not found!")
    exit()

print("Camera opened!")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand.landmark:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            np.save(os.path.join(DATA_PATH, str(count)), landmarks)
            count += 1
            print("Collecting:", gesture_name, count)

    cv2.imshow("Collecting Data", frame)

    if count >= 200:
        print("Done collecting!")
        break

cap.release()
cv2.destroyAllWindows()