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

def main():
    while capture.isOpened():
        ret,frame=capture.read()
        if not ret:
            break
        frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        faceResults=faceDetection.process(frameRGB)
        if faceResults.detections:
            for detection in faceResults.detections:
                borderBox=detection.location_data.relative_bounding_box
                rows,cols=frame.shape
                x,y,w,h=int(borderBox.xmin*cols),int(borderBox.ymin*rows),int(borderBox.width*cols),int(borderBox.height*rows)
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                




if __name__=='__main__':
    main()

