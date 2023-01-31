import cv2 as cv
import os
import numpy as np
import time



def train():
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    cap = cv.VideoCapture(0)

    # taking the name as an input
    name = str(input("Please Enter Your Name! "))
    people = [name]
    DIR = r'C:\Users\vraja\OneDrive\Documents\Tello\Faces'

    #turn on the camera and display
    t = 0
    while t<5:
        _, img = cap.read()
        cv.imshow('img',img)
        time.sleep(0.1)
        t = t+0.1
        k = cv.waitKey(30) & 0xff

    # Detecting if Face is Recognizable
    detection = 0
    while detection != 1 :
        _, img = cap.read()
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray,1.1,4)
        detection = len(faces)
        cv.imshow('img',img)
        if detection > 0:
            print("Good Job, Face Detected")
            time.sleep(3)
        else:
            print("Please Move near the webcam to Detect Face!")
        k = cv.waitKey(30) & 0xff
    cv.destroyAllWindows()
    print("Look into the Camera for 10 seconds!")
    time.sleep(3)
    t = 0
    while t < 3:
        _, img = cap.read()
        cv.imshow('Training',img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray,1.1,4)
        detection = len(faces)

        if detection ==1:
            cv.imwrite(f'Faces/{time.time()}.jpg',img)
        k = cv.waitKey(30) & 0xff
        time.sleep(0.1)
        t = t+0.1

    print("Now Move around your face Slightly in all Directions!")

    t = 0
    while t < 15:
        _, img = cap.read()
        cv.imshow('Training',img)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = haar_cascade.detectMultiScale(gray,1.1,4)
        detection = len(faces)

        if detection ==1:
            cv.imwrite(f'Faces/{time.time()}.jpg',img)
        k = cv.waitKey(30) & 0xff
        time.sleep(0.1)
        t = t+0.1

    print("Face Training Complete!")

    # training the ML model
    haar_cascade = cv.CascadeClassifier('haar_face.xml')
    # image arrays of faces
    features = []
    # corresponding labels of the faces
    labels = []

    def create_train():
        for person in people:
            path = DIR
            label = people.index(person)

            for img in os.listdir(path):
                img_path = os.path.join(path,img)

                # reading the image
                img_array = cv.imread(img_path)
                gray = cv.cvtColor(img_array,cv.COLOR_BGR2GRAY)

                faces_rect = haar_cascade.detectMultiScale(gray,scaleFactor = 1.1, minNeighbors = 4)

                for (x,y,w,h) in faces_rect:
                    faces_roi = gray[y:y+h,x:x+w]
                    features.append(faces_roi)
                    labels.append(label)

    create_train()
    print("Training Done_____________________")
    features = np.array(features, dtype = 'object')
    labels = np.array(labels)

    print('Length of the features',len(features))
    print('Length of the labels',len(labels))

    face_recognizer = cv.face.LBPHFaceRecognizer_create()

    # Train the recognizer on the features list and the labels list
    face_recognizer.train(features,labels)
    face_recognizer.write('face_trained.yml')
    np.save('features.npy',features)
    np.save('labels.npy',labels)

    features = np.load('features.npy', allow_pickle=True)
    labels = np.load('labels.npy', allow_pickle=True)

    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    face_recognizer.read('face_trained.yml')

    return people



