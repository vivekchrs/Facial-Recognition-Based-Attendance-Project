import cv2
import face_recognition
import os
import numpy as np
from datetime import datetime


path = "AttendanceImages"
Images=[]
classNames=[]
myList=os.listdir(path)
print(myList)
for cls in myList:
    curImg=cv2.imread(f'{path}/{cls}')
    Images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])
print(classNames)

def findEncodings(Images):
    encodeList=[]
    for img in Images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return (encodeList)
encodeListKnown=findEncodings(Images)
print("Encoding Complete")

def markAttendance(name):
    with open('Attendance.csv','r+') as f:
        myDataList=f.readlines()
        nameList=[]
        for line in myDataList:
            entry=line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstr=now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstr}')






cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    imgS=cv2.resize(img,(0,0),None,0.25,0.25)  # resizing img to increase speed
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrames = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrames)

    for encodeFace,faceLoc in zip(encodeCurFrame,faceCurFrames):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        faceDis= face_recognition.face_distance(encodeListKnown,encodeFace)
        print(faceDis)
        matchIndex=-1
        if (min(faceDis) < 0.5):
            matchIndex=np.argmin(faceDis)
        print(matchIndex)

        if (not(matchIndex == -1)):
            if matches[matchIndex]:

                name= classNames[matchIndex].upper()
                print(name)
                y1,x2,y2,x1=faceLoc
                y1, x2, y2, x1=y1*4,x2*4,y2*4,x1*4   # restroring the actuall dinmenssions

                cv2.rectangle(img, (x1,y1), (x2,y2), (0,255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
                markAttendance(name)

    cv2.imshow("WEBCAM",img)
    cv2.waitKey(1)

