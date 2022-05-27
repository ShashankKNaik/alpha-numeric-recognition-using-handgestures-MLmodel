import time
from unittest import result
import cv2
import sklearn
import HandTrackingModule as htm
import math
from sklearn import feature_selection, svm,tree,neighbors
import pandas as pd
import joblib

wCam, hCam = 640, 480
 
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
 
detector = htm.handDetector()

model = joblib.load('hand_model_k5_2.pkl')


confidence = 0
previous_pred=""
resultString=""

while True:
    
    success, img = cap.read()
    img = detector.findHands(img)

    x,y,w,h=200,10,200,400
    frame = cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),4)
    cv2.putText(frame,"",(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX ,1, (18,5,255), 2, cv2.LINE_AA )

    lmList = detector.findPosition(img, draw=False)
    test=[]
    pred=[""]
    if lmList!=[]:
        arr=[]

        m=lmList.pop(9)

        featureNotSelect=[1,5,13,17]

        for i in range(0,20):
            if i in featureNotSelect:
                continue
            var=1
            if lmList[i][0]<m[0]:
                var=-1
            arr.append(int(math.sqrt((lmList[i][0]-m[0])**2 + (lmList[i][1]-m[1])**2)) * var)


        test.append(arr)
        pred=model.predict(test)
        
        if(previous_pred==pred[0]):
            confidence+=1
        else:
            confidence=0
        if confidence==20:
            confidence=0
            if pred[0]=='!':
                resultString=resultString[:-1]
            else:
                resultString+=pred[0]

        previous_pred=pred[0]
    #________
    cTime = time.time()
    fps =5 
    pTime = cTime
 
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)


    
    cv2.putText(img,resultString,(10,300),cv2.FONT_HERSHEY_COMPLEX ,1, (18,5,255), 1, cv2.LINE_AA )
    cv2.putText(img,pred[0],(10,50),cv2.FONT_HERSHEY_COMPLEX ,2, (18,5,255), 2, cv2.LINE_AA )

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if(key == ord('q')):                           # press 'q' to quit
        break
print(sklearn.__version___)
cap.release()
cv2.destroyAllWindows()

