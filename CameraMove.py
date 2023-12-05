import pyautogui
import cv2
import mediapipe as mp
import numpy as np

import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

from datetime import datetime
pyautogui.FAILSAFE = False

class CameraMove:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()

    def find_faces_original(self, img):
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                min_x, min_y = w, h
                max_x, max_y = 0, 0

                for id, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    min_x, min_y = min(x, min_x), min(y, min_y)
                    max_x, max_y = max(x, max_x), max(y, max_y)

                # Allargamento dei bordi del rettangolo
                min_x = max(0, min_x - int(0.05 * w))
                max_x = min(w, max_x + int(0.05 * w))
                min_y = max(0, min_y - int(0.1 * h))
                max_y = min(h, max_y + int(0.1 * h))

                cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
                self.mp_drawing.draw_landmarks(img, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                                               self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                           circle_radius=1),
                                               self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1,
                                                                           circle_radius=1))
        return img

    def find_faces_clown(self, img):
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                min_x, min_y = w, h
                max_x, max_y = 0, 0

                for id, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    min_x, min_y = min(x, min_x), min(y, min_y)
                    max_x, max_y = max(x, max_x), max(y, max_y)

                # Calcola la larghezza del viso
                face_width = max_x - min_x

                # Calcola il raggio del cerchio in base alla larghezza del viso
                radius = int(face_width * 0.15)  # Adatta questo fattore moltiplicativo come necessario

                # Disegna un pallino rosso al naso (landmark ID 4)
                nose = face_landmarks.landmark[4]
                nose_x, nose_y = int(nose.x * w), int(nose.y * h)
                cv2.circle(img, (nose_x, nose_y), radius=radius, color=(0, 0, 255), thickness=-1)

                # Allargamento dei bordi del rettangolo
                min_x = max(0, min_x - int(0.05 * w))
                max_x = min(w, max_x + int(0.05 * w))
                min_y = max(0, min_y - int(0.1 * h))
                max_y = min(h, max_y + int(0.1 * h))

        return img

    def find_faces_black(self, img):
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                min_x, min_y = w, h
                max_x, max_y = 0, 0

                for id, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    min_x, min_y = min(x, min_x), min(y, min_y)
                    max_x, max_y = max(x, max_x), max(y, max_y)


                min_x = max(0, min_x - int(0.05 * w))
                max_x = min(w, max_x + int(0.05 * w))
                min_y = max(0, min_y - int(0.1 * h))
                max_y = min(h, max_y + int(0.1 * h))

                # Creazione di una maschera nera della stessa dimensione dell'immagine
                mask = np.zeros((h, w), dtype=np.uint8)
                # Rendere bianca l'area all'interno del rettangolo
                mask[min_y:max_y, min_x:max_x] = 255

                # Applicazione della maschera all'immagine
                img_masked = cv2.bitwise_and(img, img, mask=mask)

                self.mp_drawing.draw_landmarks(img_masked, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                                               self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                           circle_radius=1),
                                               self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1,
                                                                           circle_radius=1))
        return img_masked


    def find_faces_segmented(self, img):
        h, w, _ = img.shape
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)
        face_img_resized = None

        if results.multi_face_landmarks is not None:
            for face_landmarks in results.multi_face_landmarks:
                min_x, min_y = w, h
                max_x, max_y = 0, 0

                for id, lm in enumerate(face_landmarks.landmark):
                    x, y = int(lm.x * w), int(lm.y * h)
                    min_x, min_y = min(x, min_x), min(y, min_y)
                    max_x, max_y = max(x, max_x), max(y, max_y)

                min_x = max(0, min_x - int(0.05 * w))
                max_x = min(w, max_x + int(0.05 * w))
                min_y = max(0, min_y - int(0.1 * h))
                max_y = min(h, max_y + int(0.1 * h))

                # Estrazione dell'area del viso
                face_img = img[min_y:max_y, min_x:max_x]

                # Ridimensionamento dell'immagine del viso per adattarla alla dimensione della finestra
                face_img_resized = cv2.resize(face_img, (640, 480))

        if face_img_resized is None:
            # Visualizza uno smile rosso
            face_img_resized = cv2.putText(img, ":(", (w//2, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 7, cv2.LINE_AA)

        return face_img_resized








