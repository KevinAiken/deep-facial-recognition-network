import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
import socket
import embedding_generator
import pickle

#
# This code is mostly taken from https://github.com/shantnu/Webcam-Face-Detect
#

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
# log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
count = 0

s = socket.socket()
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
host = "0.0.0.0"
port = 5555
s.bind((host, port))
s.listen(5)
c, addr = s.accept()

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        # this if slows down the rate of calculating and sending embeddings
        if count % 30 == 0:
            facePicture = frame[y:y + h, x:x + w]
            faceEmbedding = embedding_generator.face_encodings(facePicture)
            if len(faceEmbedding) > 0:
                # print(embedding_generator.face_encodings(facePicture)[0].tolist())
                # print(type(embedding_generator.face_encodings(facePicture)[0]))

                data = str(embedding_generator.face_encodings(facePicture)[0]).replace("\n", "")
                data += "\n"

                c.send(data.encode('utf-8'))
        count += 1
    if anterior != len(faces):
        anterior = len(faces)
        # log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
