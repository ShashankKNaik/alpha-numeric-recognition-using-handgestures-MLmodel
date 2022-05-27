import time
import cv2
import HandTrackingModule as htm
import math

wCam, hCam = 640, 480
 
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
 
detector = htm.handDetector()

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    x,y,w,h=200,10,200,400
    frame = cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),4)
    cv2.putText(frame,"",(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX ,1, (18,5,255), 2, cv2.LINE_AA )

    lmList = detector.findPosition(img, draw=False)
    
    if lmList!=[]:
        arr=[]
        m=lmList.pop(9)
        for i in range(0,20):
            var=1
            if lmList[i][0]<m[0]:
                var=-1
            arr.append(int(math.sqrt((lmList[i][0]-m[0])**2 + (lmList[i][1]-m[1])**2)) * var)
        
        f = open("data.txt", "a")
        f.write("A,"+str(arr)+"\n")
    

    cTime = time.time()
    fps = 5 
    pTime = cTime
 
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    
    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if(key == ord('q')):                           # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

