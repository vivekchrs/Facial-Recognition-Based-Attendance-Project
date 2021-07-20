import cv2
import numpy as np
import face_recognition

imgElon= face_recognition.load_image_file("Images/elonMusk.jfif")
imgTest= face_recognition.load_image_file("Images/elonTest.jfif")

imgElon=cv2.cvtColor(imgElon,cv2.COLOR_BGR2RGB)
imgTest=cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(imgElon)[0]   # give us 4 values as if forming rectangle on face
#print(faceLoc)                                          # top,right,bottom,left
encodeElon=face_recognition.face_encodings(imgElon)[0]
# print(encodeElon)        # 128 point to measure distances for recognition
cv2.rectangle(imgElon,(faceLoc[3],faceLoc[0]),(faceLoc[1],faceLoc[2]),(255,0,255),2)


faceLocTest=face_recognition.face_locations(imgTest)[0]   # give us 4 values as if forming rectangle on face
#print(faceLocTest)                                          # top,right,bottom,left
encodeTest=face_recognition.face_encodings(imgTest)[0]
#print(encodeElon)
cv2.rectangle(imgTest,(faceLocTest[3],faceLocTest[0]),(faceLocTest[1],faceLocTest[2]),(255,0,255),2)

results= face_recognition.compare_faces([encodeElon],encodeTest)      # using linear svm
faceDis=face_recognition.face_distance([encodeElon],encodeTest)         # distance or how close the test is to original img (lower the dis better the match is)
print(results,faceDis)
cv2.putText(imgTest,f'{results} {round(faceDis[0],2)}',(10,10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)

cv2.imshow("elon",imgElon)
cv2.imshow("elonTest",imgTest)
cv2.waitKey(0)