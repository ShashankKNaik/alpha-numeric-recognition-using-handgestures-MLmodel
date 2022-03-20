"""
Popular algorithms that can be used for multi-class classification include:

k-Nearest Neighbors.
Decision Trees.
Naive Bayes.
Random Forest.
Gradient Boosting.
"""

import time
import cv2
import HandTrackingModule as htm
import math
from sklearn import feature_selection, svm,tree,neighbors
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelEncoder

import joblib

wCam, hCam = 640, 480
 
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

pTime = 0
 
detector = htm.handDetector()

###################################jz pt

dataset = pd.read_csv('newData.csv')
x = dataset[['0','2', '3', '4', '6', '7', '8', '10', '11', '12', '14', '15', '16', '18', '19', '20']].values
y = dataset['letter'].values
# model=svm.SVC().fit(x,y)  #clf = SVC(C=1.0, kernel='rbf').fit(X_train,y_train)
model=neighbors.KNeighborsClassifier().fit(x,y)
# model=tree.DecisionTreeClassifier().fit(x,y)

joblib.dump(model, 'hand_model.pkl')

while True:
    success, img = cap.read()
    img = detector.findHands(img)

    x,y,w,h=200,10,200,400
    frame = cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),4)
    cv2.putText(frame,"",(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX ,1, (18,5,255), 2, cv2.LINE_AA )

    lmList = detector.findPosition(img, draw=False)
    test=[]
    pred=["Nope"]
    if lmList!=[]:
        arr=[]
        # for i in range(1,21):
        #     arr.append( int(math.sqrt((lmList[i][0]-lmList[0][0])**2 + (lmList[i][1]-lmList[0][1])**2)))
        
        # f = open("data.txt", "a")
        # f.write("B,"+str(arr)+"\n")

        #_______
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


    #________
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
 
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), cv2.FONT_HERSHEY_PLAIN,
                3, (255, 0, 0), 3)
    

    cv2.putText(img,pred[0],(10,50),cv2.FONT_HERSHEY_COMPLEX ,2, (18,5,255), 2, cv2.LINE_AA )

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if(key == ord('q')):                           # press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

