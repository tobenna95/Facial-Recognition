import numpy as np
import cv2
import pickle

#Using cascades
face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_eye.xml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person_name": 1}
print(labels)
with open("labels.pickle", 'rb') as f:
    orig_labels = pickle.load(f)
    labels = {v:k for k,v in orig_labels.items()}

#To open the device at the ID 
cap = cv2.VideoCapture(0)

#To check whether user selected camera is opened successfully
if not (cap.isOpened()):
    print ('Could not open video device')

while(True):
    #To capture frame-by-frame
    ret, frame = cap.read()

    #To convert to gray for facial capture
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        # print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]  #region of interest (y-coord to y-coord-height)
        roi_color = frame[y:y+h, x:x+w]

        #recognize
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45 and conf <= 105:
            print(id_)
            print(labels[id_])
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (0, 0, 0)
            stroke = 3
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)
            

        img_item = "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (0, 0, 255) #BGR
        stroke = 3
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (255,0,0), 2)

    #To display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainer.yml")

cap.release()
cv2.destroyAllWindows()