import cv2 as cv
import mediapipe as mp


mpHands=mp.solutions.hands
mpFace=mp.solutions.face_detection
mpDrawing=mp.solutions.drawing_utils
capture=cv.VideoCapture(0)
hands=mpHands.Hands(min_detection_confidence=0.7,min_tracking_confidence=0.7)
faceDetection=mpFace.FaceDetection(min_detection_confidence=0.7)
Fingers=[4,8,12,16,20]

def counting_fingers(handlandmarks):
    """
    This will Count number of fingers raised based on hand landmarks
    """
    count=0
    if handlandmarks:
        landmarks=handlandmarks.landmark
        for tip in Fingers[1:]:
            if landmarks[tip].y<landmarks[tip-2].y:
                count+=1
        if landmarks[Fingers[0]].x > landmarks[Fingers[0]-1].x:
            count+=1
    return count

