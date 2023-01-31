import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['Raj','Dhwani']
features = np.load('features.npy', allow_pickle=True)
labels = np.load('labels.npy', allow_pickle=True)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

img = cv.imread(r'C:\Users\vraja\OneDrive\Documents\OpenCV Course\Faces_validation\Raj\r1.jpg')
gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
# cv.imshow('Person',gray)

# detect face in the image
faces_rect = haar_cascade.detectMultiScale(gray,1.1,7)

for (x,y,w,h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]

    label,confidence = face_recognizer.predict(faces_roi)
    print(f'Label = {people[label]} with a confidence of {confidence}')
    cv.putText(img,str(people[label]+" /Confidence: "+str(confidence)),(400,200),cv.FONT_HERSHEY_COMPLEX,1.0,(255,255,255),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)

img = cv.resize(img,(img.shape[1]//2,img.shape[0]//2))
cv.imshow('Detected Face',img)

cv.waitKey(0)