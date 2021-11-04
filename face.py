import face_recognition
import numpy as np
import cv2 as cv
import pickle

#import eos-pyp
image = face_recognition.load_image_file(r"C:\Users\oluwa\Downloads\Telegram Desktop\face.jpg")
face_locations = face_recognition.face_locations(image)
#face_recognition.face_landmarks(face_image=image, face_locations=face_locations, model=)
#print(face_locations)
known_image = face_recognition.load_image_file(r"C:\Users\oluwa\Downloads\Telegram Desktop\face.jpg")
unknown_image = face_recognition.load_image_file(r"C:\Users\oluwa\Downloads\Telegram Desktop\shephy.jpg")
shephy_encoding = face_recognition.face_encodings(known_image)[0]
unkown_encoding = face_recognition.face_encodings(unknown_image)[0]
results = face_recognition.compare_faces([shephy_encoding], unkown_encoding)
#print(results)


face_cascade = cv.CascadeClassifier(r'C:\Users\oluwa\Documents\passport\cascades\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv.CascadeClassifier(r'C:\Users\oluwa\OneDrive\Documents\projects\passport\cascade\data\haarcascade_eye.xml')
smile_cascade = cv.CascadeClassifier(r'C:\Users\oluwa\OneDrive\Documents\projects\passport\cascade\data\haarcascade_smile.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person's name: 1"}
with open("labels.pkl", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}


cap = cv.VideoCapture(0)
while (True):
      #capture frame by fame
      ret, frame = cap.read()
      #convert to gray
      gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
      faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
      for (x,y,w,h) in faces:
            #print(x,y,w,h)
            #roi means region of interest
            roi_gray = gray[y:y+h, x:x+w]# these mean coordinates, x1:x2, y1:y2
            roi_color = frame[y:y+h, x:x+w]

            #recognizer is a deep learning model
            id_, conf = recognizer.predict(roi_gray)#this will give us the label back and the confidence
            #noticed the confidence is above 100, hence the function below
            if conf >=45 and conf <=85:
                  print(id_)
                  print(labels[id_])
                  font = cv.FONT_HERSHEY_COMPLEX
                  name = labels[id_]
                  color = (255, 255, 255)
                  stroke = 2
                  cv.putText(frame, name, (x,y), font, 1, color, stroke,cv.LINE_AA)

            


            cv.imwrite("face_img.png", roi_gray)
            color = (255,0,0) #BGR
            stroke = 2 #this means how thick we want the lines to be
            width = x + w
            height = y + h
            cv.rectangle(frame, (x,y), (width,height), color, stroke)
            sub_items = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.5, minNeighbors=5)
            for (ex,ey,ew,eh) in sub_items:
                  cv.rectangle(roi_color, (ex,ey), (ex+ew, ey+eh), (0,255,0), 2)


      #display
      cv.imshow('frame', frame)
      if cv.waitKey(20) & 0xFF == ord('q'):
            break
      





cap.release()
cv.destroyAllWindows()
