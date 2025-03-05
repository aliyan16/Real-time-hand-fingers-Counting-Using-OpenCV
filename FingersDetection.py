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
                rows,cols,_=frame.shape
                x,y,w,h=int(borderBox.xmin*cols),int(borderBox.ymin*rows),int(borderBox.width*cols),int(borderBox.height*rows)
                cv.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        handResults=hands.process(frameRGB)
        if handResults.multi_hand_landmarks:
            for handLandmarks in handResults.multi_hand_landmarks:
                xmin,ymin,xmax,ymax=cols,rows,0,0
                for landmrk in handLandmarks.landmark:
                    x,y=int(landmrk.x*cols),int(landmrk.y*rows)
                    xmin,ymin=min(x,xmin),min(y,ymin)
                    xmax,ymax=max(x,xmax),max(y,ymax)
                cv.rectangle(frame,(xmin,ymin),(xmax,ymax),(255,0,0),2)
                fingerCount=counting_fingers(handLandmarks)
                cv.putText(frame,f'Fingers:{fingerCount}',(xmin,ymin-10),cv.FONT_ITALIC,1,(255,255,255),2)
                mpDrawing.draw_landmarks(frame,handLandmarks,mpHands.HAND_CONNECTIONS)
        cv.imshow('Face and Hand Detection',frame)
        if cv.waitKey(1) & 0xFF==ord('q'):
            break
    capture.release()
    cv.destroyAllWindows()




if __name__=='__main__':
    main()

