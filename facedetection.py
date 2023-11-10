#  FACE DETECTION IN IMAGE STEP-1
import cv2
import mediapipe as mp
import time


cap=cv2.VideoCapture("1.mp4")
pTime=0
mpFaceDetection=mp.solutions.face_detection
mp_Draw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection(0.75)

while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    result=faceDetection.process(imgRGB)
    print(result)

    if result.detections:
        for id,detections in enumerate(result.detections):
            #mp_Draw.draw_detection(img,detections)
          #  print(id,detections)
          #  print(detections.location_data.relative_bounding_box)
            bboxC=detections.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=int(bboxC.xmin*iw),int(bboxC.ymin*ih),\
                 int(bboxC.width*iw),int(bboxC.height*iw+50)
            # cv2.rectangle(img,bbox,(255,0,255),2)
            cv2.putText(img,f'{int(detections.score[0]*100)}%',(bbox[0],bbox[1]-20),cv2.FONT_HERSHEY_PLAIN,3,(255,0,255),2)
            x,y,w,h=bbox
            l,t=30,5
            x1,y1=x+w,y+h
            cv2.rectangle(img,bbox,(255,0,255),1)
            #Top left corner
            cv2.line(img,(x,y),(x+l,y),(255,0,255),t)
            cv2.line(img,(x,y),(x,y+l),(255,0,255),t)
            #Top right corner
            cv2.line(img,(x1,y),(x1-l,y),(255,0,255),t)
            cv2.line(img,(x1,y),(x1,y+l),(255,0,255),t)
            #Bottom left corner
            cv2.line(img,(x,y1),(x+l,y1),(255,0,255),t)
            cv2.line(img,(x,y1),(x,y1-l),(255,0,255),t)
            #Top right corner
            cv2.line(img,(x1,y1),(x1-l,y1),(255,0,255),t)
            cv2.line(img,(x1,y1),(x1,y1-l),(255,0,255),t)
        
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv2.putText(img,f'FPS:{int(fps)}',(28,70),cv2.FONT_HERSHEY_PLAIN,3,(255,0,0),2)
    cv2.imshow("img",img)
    cv2.waitKey(30)

