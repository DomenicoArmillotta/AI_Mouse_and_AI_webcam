import cv2
from EyeTracker import EyeTracker
from handTracker import handTracker
from Tracker import Tracker
from CameraMove import CameraMove
from PianoHand import PianoHand


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
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 'S' key to stop
            break

    cap.release()
    cv2.destroyAllWindows()
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
        if cv2.waitKey(1) & 0xFF == ord('s'):  # Press 'S' to stop
            break  # Interrompe il ciclo while

    cap.release()
    cv2.destroyAllWindows()
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
        if cv2.waitKey(1) & 0xFF == ord('s'):  # stop webcam acquisition with 'S' key
            break
    cap.release()
    cv2.destroyAllWindows()
    print(tracker.df)
    tracker.plot_data()
    tracker.plot_ear()

def main_webcam():
    cap = cv2.VideoCapture(0)
    detector = CameraMove()

    while True:
        success, image = cap.read()
        # image = detector.find_faces_original(image)
        image = detector.find_faces_laser(image)
        # image = detector.find_faces_black(image)
        #image = detector.find_faces_segmented(image)
        # image = detector.find_faces_clown(image)
        cv2.imshow("Video", image)
        image.flags.writeable = False
        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()


def main_piano_hand():
    cap = cv2.VideoCapture(0)
    piano_hand = PianoHand()

    while True:
        success, image = cap.read()

        # Rileva le mani e disegna il piano
        piano_hand.detect_hands(image)

        # Visualizza l'immagine con il piano sovrapposto
        cv2.imshow("Piano Hand", image)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main_piano_hand()
    # uncomment the favourite main
    # hand tracking + eye blink tracking
    #main()
    # only eye tracking with plot
    # main_eye()
    # only hand tracking with angle detection enabled
    #main_finger()
    main_webcam()
