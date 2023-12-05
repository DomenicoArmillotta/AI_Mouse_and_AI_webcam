# ___________________________________PROVA with Contours_____________________-
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance
import time


class EyeTracker:
    def __init__(self):
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.draw_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.df = pd.DataFrame(columns=['timestamp', 'ear_left', 'ear_right'])

    def convert_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def process_frame(self, image):
        return self.face_mesh.process(image)

    def draw_landmarks(self, image, results):
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(image, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                                               self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                           circle_radius=1),
                                               self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1,
                                                                           circle_radius=1))

                ear_left = self.calculate_ear(face_landmarks, [362, 385, 387, 263, 373, 380])
                ear_right = self.calculate_ear(face_landmarks, [33, 160, 158, 133, 153, 144])

                # Aggiungi il tuo valore di soglia qui
                threshold = 0.35
                if ear_left < threshold:
                    cv2.putText(image, 'Left', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if ear_right < threshold:
                    cv2.putText(image, 'Right', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def calculate_ear(self, landmarks, eye_indices):
        p1, p2, p3, p4, p5, p6 = [landmarks.landmark[i] for i in eye_indices]
        ear = (distance.euclidean((p2.x, p2.y), (p6.x, p6.y)) +
               distance.euclidean((p3.x, p3.y), (p5.x, p5.y))) / (
                      2 * distance.euclidean((p1.x, p1.y), (p4.x, p4.y)))
        return ear

    def update_dataframe(self, ear_left, ear_right):
        new_df = pd.DataFrame({'timestamp': [pd.Timestamp.now()],
                               'ear_left': [ear_left],
                               'ear_right': [ear_right]})
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def plot_ear(self):
        plt.plot(self.df['timestamp'], self.df['ear_left'], label='Left EAR')
        plt.plot(self.df['timestamp'], self.df['ear_right'], label='Right EAR')
        plt.legend()
        plt.show()




