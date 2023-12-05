import cv2
import mediapipe as mp
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from EyeTracker import EyeTracker
from handTracker import handTracker
from Tracker import Tracker
def main_eye():
    cap = cv2.VideoCapture(0)
    eye_tracker = EyeTracker()

    while True:
        success, image = cap.read()
        image.flags.writeable = False
        rgb_image = eye_tracker.convert_to_rgb(image)
        results = eye_tracker.process_frame(rgb_image)
        image.flags.writeable = True
        eye_tracker.draw_landmarks(image, results)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                ear_left = eye_tracker.calculate_ear(face_landmarks, [362, 385, 387, 263, 373, 380])
                ear_right = eye_tracker.calculate_ear(face_landmarks, [33, 160, 158, 133, 153, 144])
                eye_tracker.update_dataframe(ear_left, ear_right)

        cv2.imshow("Video", image)
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Se il tasto 's' viene premuto
            break  # Interrompe il ciclo while

    cap.release()  # Rilascia l'oggetto di acquisizione video
    cv2.destroyAllWindows()  # Chiude tutte le finestre create da OpenCV
    eye_tracker.plot_ear()


def main_finger():
    cap = cv2.VideoCapture(0)
    tracker = handTracker()

    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        tracker.positionFinderMouseClick(image)
        '''
        lmList = 
        if len(lmList) != 0:
            for item in lmList:
                print(item)
        '''


        cv2.imshow("Video", image)
        image.flags.writeable = False
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Se il tasto 's' viene premuto
            break  # Interrompe il ciclo while

    cap.release()  # Rilascia l'oggetto di acquisizione video
    cv2.destroyAllWindows()  # Chiude tutte le finestre create da OpenCV
    print(tracker.data)
    tracker.plot_data()
    # plotted only the smoothed data
    tracker.smooth_and_plot()

def main():
    cap = cv2.VideoCapture(0)
    tracker = Tracker()
    while True:
        success, image = cap.read()
        image = tracker.handsFinder(image)
        tracker.positionFinderMouseClickWithFace(image)
        cv2.imshow("Video", image)
        image.flags.writeable = False
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Se il tasto 's' viene premuto
            break  # Interrompe il ciclo while
    cap.release()  # Rilascia l'oggetto di acquisizione video
    cv2.destroyAllWindows()  # Chiude tutte le finestre create da OpenCV
    print(tracker.df)
    tracker.plot_data()
    tracker.plot_ear()



if __name__ == "__main__":
    main()