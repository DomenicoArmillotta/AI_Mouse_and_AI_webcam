import cv2
import mediapipe as mp
import numpy as np

class PianoHand:
    def __init__(self):
        # Inizializza la libreria Mediapipe per la detection delle mani
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands()

        # Configura il piano a 61 tasti
        self.piano = np.zeros((480, 640, 3), dtype=np.uint8)  # Dimensioni arbitrarie, puoi adattarle

    def detect_hands(self, image):
        # Converti l'immagine in scala di grigi in un'immagine a colori
        color_image = image.copy()

        # Esegui la detection delle mani
        results = self.hands.process(color_image)

        # Disegna il piano a 32 tasti sull'immagine
        self.draw_piano(color_image)

        # Verifica se sono state rilevate mani
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Disegna un pallino di colore diverso su ciascuna punta del dito
                for landmark in self.mp_hands.HandLandmark:
                    x, y = int(hand_landmarks.landmark[landmark].x * 640), int(
                        hand_landmarks.landmark[landmark].y * 480)
                    finger_tip_color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
                    cv2.circle(color_image, (x, y), 10, finger_tip_color, -1)

    def draw_piano(self, image):
        # Dimensioni e posizione del piano
        piano_height = int(0.3 * image.shape[0])  # 30% inferiore
        piano_width = image.shape[1]
        piano_y = image.shape[0] - piano_height
        key_width = piano_width // 32  # Dividi il piano in 32 tasti

        # Colore dei tasti
        white_key_color = (255, 255, 255)
        black_key_color = (0, 0, 0)

        # Trasparenza dei tasti
        alpha = 0.8

        # Creazione di un'immagine nera del piano
        piano_image = np.zeros_like(image)

        # Disegna i tasti bianchi
        for i in range(32):
            x = i * key_width
            y = piano_y
            cv2.rectangle(piano_image, (x, y), (x + key_width, y + piano_height), white_key_color, -1)

        # Disegna i tasti neri (ogni secondo tasto)
        for i in range(1, 32, 2):
            x = i * key_width
            y = piano_y
            cv2.rectangle(piano_image, (x, y), (x + key_width, y + piano_height // 2), black_key_color, -1)

        # Combina l'immagine del piano con l'immagine originale
        combined_image = cv2.addWeighted(image, 1, piano_image, alpha, 0)

        # Disegna il contorno dei tasti sull'immagine combinata
        for i in range(32):
            x = i * key_width
            y = piano_y
            cv2.rectangle(combined_image, (x, y), (x + key_width, y + piano_height), white_key_color, 1)  # Contorno

        for i in range(1, 32, 2):
            x = i * key_width
            y = piano_y
            cv2.rectangle(combined_image, (x, y), (x + key_width, y + piano_height // 2), black_key_color,
                          1)  # Contorno

        # Copia l'immagine combinata nell'immagine originale
        image[:, :] = combined_image
