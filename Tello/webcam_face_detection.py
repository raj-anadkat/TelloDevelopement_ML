import cv2
face_cascade = cv2.CascadeClassifier('haar_face.xml')
cap = cv2.VideoCapture(0)


while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.1,4)

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w+8,y+h+20),(0,0,255),2)
        area = h*w
        if area < 40000 and area > 15000:
            print("Perfect!")
        elif area > 40000:
            print("Too close!")
        
        else:
            print("Too far!")
    cv2.imshow('img',img)

    k = cv2.waitKey(30) & 0xff

#cap.release()


