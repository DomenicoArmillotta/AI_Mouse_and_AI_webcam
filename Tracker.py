import pyautogui
import cv2
import mediapipe as mp
import math
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance

from datetime import datetime
pyautogui.FAILSAFE = False

#in this implementation is disabled the computation referred to angle x and y of the hand to velocize the computation
class Tracker() :
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode  # to set the input asvideo stream
        self.maxHands = maxHands  # setted as 2 hand
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.previous_positions = []  # to draw a line with prev position
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.draw_spec = self.mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
        self.df = pd.DataFrame(columns=['timestamp', 'centroid_x', 'centroid_y' , 'ear_left','ear_right'])

    def convert_to_rgb(self, image):
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def handsFinder(self, image, draw=True):
        imageRGB = self.convert_to_rgb(image)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def process_frame(self, image):
        return self.face_mesh.process(image)

    def calculate_ear(self, landmarks, eye_indices):
        p1, p2, p3, p4, p5, p6 = [landmarks.landmark[i] for i in eye_indices]
        ear = (distance.euclidean((p2.x, p2.y), (p6.x, p6.y)) +
               distance.euclidean((p3.x, p3.y), (p5.x, p5.y))) / (
                      2 * distance.euclidean((p1.x, p1.y), (p4.x, p4.y)))
        return ear

    # list of all finger with x,y and the angles of hand
    def positionFinderMouseClickWithFace(self, image, handNo=0, draw=True):
        lmlist = []
        h, w, _ = image.shape
        border_size_x = int(w * 0.1)
        border_size_y = int(h * 0.1)
        prev_centroid_x = 0
        prev_centroid_y = 0
        centroid_x = 0  # Initialize outside the loop
        centroid_y = 0
        prev_centroid_x, prev_centroid_y = 0, 0
        if self.results.multi_hand_landmarks:
            for handNo, Hand in enumerate(self.results.multi_hand_landmarks):
                sum_x = 0
                sum_y = 0
                for id, lm in enumerate(Hand.landmark):
                    h, w, c = image.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    sum_x += cx
                    sum_y += cy
                    lmlist.append([id, cx, cy])
                    if draw == True and id == 8:
                        cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # Pallino rosa sul indice
                        mouse_x, mouse_y = pyautogui.position()

                        pyautogui.moveTo(1980 - cx * 3, cy * 2.20)
                centroid_x = sum_x // len(Hand.landmark)
                centroid_y = sum_y // len(Hand.landmark)
                if draw:
                    cv2.circle(image, (centroid_x, centroid_y), 15, (0, 0, 255),
                               cv2.FILLED)  # Pallino rosso al centro della mano

                lmlist.append(['center', centroid_x, centroid_y])

                self.previous_positions.append((centroid_x, centroid_y))

                if len(self.previous_positions) > 10:
                    self.previous_positions.pop(0)

                for i in range(len(self.previous_positions) - 1):
                    cv2.line(image, self.previous_positions[i], self.previous_positions[i + 1], (255, 0, 0), 2)

                # compute the distance between the finger to understand if it is closed
                distances = []
                for item in lmlist:
                    if len(item) == 3:  # if the element is a triple , because is saved also the data referred to angles
                        id, cx, cy = item
                        if id in [4, 12, 16, 20]:  # specific finger
                            distance = ((cx - centroid_x) ** 2 + (cy - centroid_y) ** 2) ** 0.5
                            distances.append(distance)

                # threshold to determine if a hand is closed
                threshold = 60
                closed_fingers = [d > threshold for d in distances]

                if all(closed for i, closed in enumerate(closed_fingers) if i == 1):
                    cv2.putText(image, 'OPEN', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    #pyautogui.click()
        results = self.process_frame(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.mp_drawing.draw_landmarks(image, face_landmarks, self.mp_face_mesh.FACEMESH_TESSELATION,
                                               self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1,
                                                                           circle_radius=1),
                                               self.mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1,
                                                                           circle_radius=1))
                ear_left = self.calculate_ear(face_landmarks, [362, 385, 387, 263, 373, 380])
                ear_right = self.calculate_ear(face_landmarks, [33, 160, 158, 133, 153, 144])

                # Threshold for eye blink detection
                threshold = 0.33
                if (ear_left < threshold) and (ear_right < threshold) :
                    cv2.putText(image, 'Left', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, 'Right', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # pyautogui.click()
                if ear_left < threshold:
                    cv2.putText(image, 'Left', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                if ear_right < threshold:
                    cv2.putText(image, 'Right', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                self.update_dataframe(centroid_x, centroid_y, ear_left, ear_right)




    def update_dataframe(self,centroid_x,centroid_y, ear_left, ear_right):
        new_df = pd.DataFrame({'timestamp': [pd.Timestamp.now()],
                               'ear_left': [ear_left],
                               'centroid_x': [centroid_x],
                               'centroid_y': [centroid_y],
                               'ear_right': [ear_right]})
        self.df = pd.concat([self.df, new_df], ignore_index=True)

    def plot_data(self):
        h = 480
        w = 640
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(self.df['timestamp'], self.df['centroid_x'])
        plt.title('Centroid X over Time')
        plt.ylim([0, h])  # Imposta i limiti dell'asse y

        plt.subplot(2, 2, 2)
        plt.plot(self.df['timestamp'], self.df['centroid_y'])
        plt.title('Centroid Y over Time')
        plt.ylim([0, w])

        plt.tight_layout()
        plt.show()

    def plot_ear(self):
        plt.plot(self.df['timestamp'], self.df['ear_left'], label='Left EAR')
        plt.plot(self.df['timestamp'], self.df['ear_right'], label='Right EAR')
        plt.legend()
        plt.show()


