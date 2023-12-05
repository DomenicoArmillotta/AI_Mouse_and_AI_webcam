import pyautogui
import cv2
import mediapipe as mp
import math
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
pyautogui.FAILSAFE = False


class handTracker():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, modelComplexity=1, trackCon=0.5):
        self.mode = mode  # to sei the input asvideo stream
        self.maxHands = maxHands  # setted as 2 hand
        self.detectionCon = detectionCon
        self.modelComplex = modelComplexity
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplex,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
        self.previous_positions = []  # Aggiungi questa linea per memorizzare le posizioni precedenti
        self.data = pd.DataFrame(columns=['timestamp', 'angle_x', 'angle_y', 'centroid_x', 'centroid_y'])

    def handsFinder(self, image, draw=True):
        imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imageRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:

                if draw:
                    self.mpDraw.draw_landmarks(image, handLms, self.mpHands.HAND_CONNECTIONS)
        return image

    def calculate_rotation_angle_x(self, landmarks):
        # Verifica se i landmark necessari sono disponibili
        if len(landmarks) < 18:
            return None

        # Calcola i vettori tra i landmark per l'asse X
        vector_1_x = [landmarks[5].x - landmarks[0].x, landmarks[5].y - landmarks[0].y]
        vector_2_x = [landmarks[17].x - landmarks[0].x, landmarks[17].y - landmarks[0].y]

        # Calcola l'angolo tra i vettori per l'asse X
        angle_rad_x = math.atan2(vector_2_x[1], vector_2_x[0]) - math.atan2(vector_1_x[1], vector_1_x[0])
        angle_deg_x = math.degrees(angle_rad_x)

        # Normalizza l'angolo nell'intervallo [0, 360] per l'asse X
        if angle_deg_x < 0:
            angle_deg_x += 360

        return angle_deg_x

    def calculate_rotation_angle_y(self, landmarks):
        # Verifica se i landmark necessari sono disponibili
        if len(landmarks) < 18:
            return None

        # Calcola i vettori tra i landmark per l'asse Y
        vector_1_y = [landmarks[9].x - landmarks[0].x, landmarks[9].y - landmarks[0].y]
        vector_2_y = [landmarks[0].x - landmarks[0].x, landmarks[0].y - landmarks[0].y]

        # Calcola l'angolo tra i vettori per l'asse Y
        angle_rad_y = math.atan2(vector_2_y[1], vector_2_y[0]) - math.atan2(vector_1_y[1], vector_1_y[0])
        angle_deg_y = math.degrees(angle_rad_y)

        # Normalizza l'angolo nell'intervallo [0, 360] per l'asse Y
        if angle_deg_y < 0:
            angle_deg_y += 360

        return angle_deg_y

    # list of all finger with x,y and the angles of hand
    def positionFinder3(self, image, handNo=0, draw=True):
        lmlist = []
        if self.results.multi_hand_landmarks:
            # itero sulle mano rilevate
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
                        cv2.circle(image, (cx, cy), 15, (255, 0, 255), cv2.FILLED)  # Pallino rosa sul mignolo

                centroid_x = sum_x // len(Hand.landmark)
                centroid_y = sum_y // len(Hand.landmark)
                if draw:
                    cv2.circle(image, (centroid_x, centroid_y), 15, (0, 0, 255),
                               cv2.FILLED)  # Pallino rosso al centro della mano
                lmlist.append(['center', centroid_x, centroid_y])

                # Aggiungi la posizione corrente alla lista delle posizioni precedenti
                self.previous_positions.append((centroid_x, centroid_y))

                # Limita la lunghezza della lista delle posizioni precedenti a 10
                if len(self.previous_positions) > 10:
                    self.previous_positions.pop(0)

                # Disegna una linea tra ogni coppia di posizioni consecutive
                for i in range(len(self.previous_positions) - 1):
                    cv2.line(image, self.previous_positions[i], self.previous_positions[i + 1], (255, 0, 0), 2)

                # Calcola e aggiungi gli angoli alla lista
                angle_x = self.calculate_rotation_angle_x(Hand.landmark)
                angle_y = self.calculate_rotation_angle_y(Hand.landmark)
                lmlist.append(['angle_x', angle_x])
                lmlist.append(['angle_y', angle_y])

                # Visualizza gli angoli di rotazione sull'immagine
                cv2.putText(image, 'Angolo X: {:.2f}'.format(angle_x), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Angolo Y: {:.2f}'.format(angle_y), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                # Aggiungi le variabili al DataFrame con un timestamp
                new_data = pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'angle_x': [angle_x],
                    'angle_y': [angle_y],
                    'centroid_x': [centroid_x],
                    'centroid_y': [centroid_y]
                })
                self.data = pd.concat([self.data, new_data], ignore_index=True)
        return lmlist

    # list of all finger with x,y and the angles of hand
    def positionFinderMouse(self, image, handNo=0, draw=True):
        lmlist = []
        prev_centroid_x = 0
        prev_centroid_y = 0
        prev_centroid_x, prev_centroid_y = 0, 0  # Coordinate precedenti dell'indice
        if self.results.multi_hand_landmarks:
            # itero sulle mano rilevate
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
                        # Sposta il puntatore del mouse in base alla differenza calcolata
                        pyautogui.moveTo(1980 - cx * 3, cy * 2.20)
                centroid_x = sum_x // len(Hand.landmark)
                centroid_y = sum_y // len(Hand.landmark)
                if draw:
                    cv2.circle(image, (centroid_x, centroid_y), 15, (0, 0, 255),
                               cv2.FILLED)  # Pallino rosso al centro della mano

                lmlist.append(['center', centroid_x, centroid_y])

                # Aggiungi la posizione corrente alla lista delle posizioni precedenti
                self.previous_positions.append((centroid_x, centroid_y))

                # Limita la lunghezza della lista delle posizioni precedenti a 10
                if len(self.previous_positions) > 10:
                    self.previous_positions.pop(0)

                # Disegna una linea tra ogni coppia di posizioni consecutive
                for i in range(len(self.previous_positions) - 1):
                    cv2.line(image, self.previous_positions[i], self.previous_positions[i + 1], (255, 0, 0), 2)

                # Calcola e aggiungi gli angoli alla lista
                angle_x = self.calculate_rotation_angle_x(Hand.landmark)
                angle_y = self.calculate_rotation_angle_y(Hand.landmark)
                lmlist.append(['angle_x', angle_x])
                lmlist.append(['angle_y', angle_y])

                # Visualizza gli angoli di rotazione sull'immagine
                cv2.putText(image, 'Angolo X: {:.2f}'.format(angle_x), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Angolo Y: {:.2f}'.format(angle_y), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                # Aggiungi le variabili al DataFrame con un timestamp
                new_data = pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'angle_x': [angle_x],
                    'angle_y': [angle_y],
                    'centroid_x': [centroid_x],
                    'centroid_y': [centroid_y]
                })
                self.data = pd.concat([self.data, new_data], ignore_index=True)
        return lmlist

    # list of all finger with x,y and the angles of hand
    def positionFinderMouseClick(self, image, handNo=0, draw=True):
        lmlist = []
        h, w, _ = image.shape
        border_size_x = int(w * 0.1)
        border_size_y = int(h * 0.1)
        prev_centroid_x = 0
        prev_centroid_y = 0
        prev_centroid_x, prev_centroid_y = 0, 0  # Coordinate precedenti dell'indice
        if self.results.multi_hand_landmarks:
            # itero sulle mano rilevate
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

                        # Sposta il puntatore del mouse in base alla differenza calcolata
                        pyautogui.moveTo(1980 - cx * 3, cy * 2.20)
                centroid_x = sum_x // len(Hand.landmark)
                centroid_y = sum_y // len(Hand.landmark)
                if draw:
                    cv2.circle(image, (centroid_x, centroid_y), 15, (0, 0, 255),
                               cv2.FILLED)  # Pallino rosso al centro della mano

                lmlist.append(['center', centroid_x, centroid_y])

                # Aggiungi la posizione corrente alla lista delle posizioni precedenti
                self.previous_positions.append((centroid_x, centroid_y))

                # Limita la lunghezza della lista delle posizioni precedenti a 10
                if len(self.previous_positions) > 10:
                    self.previous_positions.pop(0)

                # Disegna una linea tra ogni coppia di posizioni consecutive
                for i in range(len(self.previous_positions) - 1):
                    cv2.line(image, self.previous_positions[i], self.previous_positions[i + 1], (255, 0, 0), 2)

                # Calcola e aggiungi gli angoli alla lista
                angle_x = self.calculate_rotation_angle_x(Hand.landmark)
                angle_y = self.calculate_rotation_angle_y(Hand.landmark)
                lmlist.append(['angle_x', angle_x])
                lmlist.append(['angle_y', angle_y])

                # Visualizza gli angoli di rotazione sull'immagine
                cv2.putText(image, 'Angolo X: {:.2f}'.format(angle_x), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(image, 'Angolo Y: {:.2f}'.format(angle_y), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255), 2, cv2.LINE_AA)
                # Aggiungi le variabili al DataFrame con un timestamp
                new_data = pd.DataFrame({
                    'timestamp': [datetime.now()],
                    'angle_x': [angle_x],
                    'angle_y': [angle_y],
                    'centroid_x': [centroid_x],
                    'centroid_y': [centroid_y]
                })
                self.data = pd.concat([self.data, new_data], ignore_index=True)

                # Calcola la distanza tra le punte delle dita e il centro della mano
                distances = []
                for item in lmlist:
                    if len(item) == 3:  # Se l'elemento è una tripla
                        id, cx, cy = item
                        if id in [4, 12, 16, 20]:  # Punte delle dita
                            distance = ((cx - centroid_x) ** 2 + (cy - centroid_y) ** 2) ** 0.5
                            distances.append(distance)

                # Soglia per determinare se una dita è chiusa o aperta
                threshold = 60
                closed_fingers = [d > threshold for d in distances]

                # Se tutte le dita tranne l'indice (il secondo elemento della lista) sono aperte
                if all(closed for i, closed in enumerate(closed_fingers) if i == 1):
                    cv2.putText(image, 'OPEN', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                    #pyautogui.click()

        #return lmlist

    def plot_data(self):
        h = 480
        w = 640

        # Dopo aver raccolto i dati, traccia i grafici
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(self.data['timestamp'], self.data['angle_x'])
        plt.title('Angle X over Time')
        plt.ylim([0, 360])  # Imposta i limiti dell'asse y

        plt.subplot(2, 2, 2)
        plt.plot(self.data['timestamp'], self.data['angle_y'])
        plt.title('Angle Y over Time')
        plt.ylim([0, 360])  # Imposta i limiti dell'asse y

        plt.subplot(2, 2, 3)
        plt.plot(self.data['timestamp'], self.data['centroid_x'])
        plt.title('Centroid X over Time')
        plt.ylim([0, h])  # Imposta i limiti dell'asse y

        plt.subplot(2, 2, 4)
        plt.plot(self.data['timestamp'], self.data['centroid_y'])
        plt.title('Centroid Y over Time')
        plt.ylim([0, w])  # Imposta i limiti dell'asse y

        plt.tight_layout()
        plt.show()

    def smooth_and_plot(self):
        h = 480
        w = 640
        # Definisci la dimensione della finestra e la deviazione standard per la gaussiana
        window_size = 5
        std_dev = 0.5

        # Crea un nuovo dataframe con le colonne smoothed
        smooth_data = pd.DataFrame()
        smooth_data['timestamp'] = self.data['timestamp']
        smooth_data['angle_x_smooth'] = self.data['angle_x'].rolling(window=window_size, win_type='gaussian',
                                                                     center=True).mean(std=std_dev)
        smooth_data['angle_y_smooth'] = self.data['angle_y'].rolling(window=window_size, win_type='gaussian',
                                                                     center=True).mean(std=std_dev)
        smooth_data['centroid_x_smooth'] = self.data['centroid_x'].rolling(window=window_size, win_type='gaussian',
                                                                           center=True).mean(std=std_dev)
        smooth_data['centroid_y_smooth'] = self.data['centroid_y'].rolling(window=window_size, win_type='gaussian',
                                                                           center=True).mean(std=std_dev)
        # Dopo aver raccolto i dati, traccia i grafici
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.plot(smooth_data['timestamp'], smooth_data['angle_x_smooth'])
        plt.title('Angle X over Time')
        plt.ylim([0, 360])  # Imposta i limiti dell'asse y

        plt.subplot(2, 2, 2)
        plt.plot(smooth_data['timestamp'], smooth_data['angle_y_smooth'])
        plt.title('Angle Y over Time')
        plt.ylim([0, 360])  # Imposta i limiti dell'asse y

        plt.subplot(2, 2, 3)
        plt.plot(smooth_data['timestamp'], smooth_data['centroid_x_smooth'])
        plt.title('Centroid X over Time')
        plt.ylim([0, h])  # Imposta i limiti dell'asse y

        plt.subplot(2, 2, 4)
        plt.plot(smooth_data['timestamp'], smooth_data['centroid_y_smooth'])
        plt.title('Centroid Y over Time')
        plt.ylim([0, w])  # Imposta i limiti dell'asse y

        plt.tight_layout()
        plt.show()




