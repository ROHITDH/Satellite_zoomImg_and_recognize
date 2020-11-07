# import the necessary packages
from __future__ import print_function
from imutils.video import VideoStream
from pyimagesearch.centroidtracker import CentroidTracker
from imutils.object_detection import non_max_suppression
from imutils import paths
from firebase import firebase
import numpy as np
import argparse
import imutils
import cv2
import time
import os



# initialize the HOG descriptor/person detector
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
firebase = firebase.FirebaseApplication('https://satell-16fa1.firebaseio.com/')
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
    help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
    help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
    help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
(H, W) = (None, None)

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])


# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
#faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(5.0)


while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    

     #///////////////////////////////////////face////////////////////////////////////////////  
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    # construct a blob from the frame, pass it through the network,
    # obtain our output predictions, and initialize the list of
    # bounding box rectangles
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H),
        (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    rects = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # filter out weak detections by ensuring the predicted
        # probability is greater than a minimum threshold
        if detections[0, 0, i, 2] > args["confidence"]:
            # compute the (x, y)-coordinates of the bounding box for
            # the object, then update the bounding box rectangles list
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            rects.append(box.astype("int"))

            # draw a bounding box surrounding the object so we can
            # visualize it
            (startX, startY, endX, endY) = box.astype("int")
            cv2.rectangle(frame, (startX, startY), (endX, endY),
                (0, 255, 0), 2)
            print("face x axis = ",startX)
            print("face y axis = ", startY)
            result = firebase.post('satell', {'FACE X axis':str(startX),'FACE Y axis':str(startY)})
           

            

    # update our centroid tracker using the computed set of bounding
    # box rectangles
    objects = ct.update(rects)

    # loop over the tracked objects
    for (objectID, centroid) in objects.items():
        # draw both the ID of the object and the centroid of the
        # object on the output frame
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    
    #////////////////////////////////////////////////////////////////////////////////////////
     # detect people in the image
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4, 4),
    padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            

    # apply non-maxima suppression to the bounding boxes using a
    # fairly large overlap threshold to try to maintain overlapping
    # boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
            cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
            print("full body x axis = ",xA)
            print("full body y axis = ",yA)
            print("  ")
            result = firebase.post('satell', {'BODY X axis':str(xA),'BODY Y axis':str(yA)})
           
            cropped = frame[yA:yB, xA:xB]
            cropped = imutils.resize(cropped, width=min(400, cropped.shape[1]))
            cropped = imutils.resize(cropped, width=200)
            
            
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read('trainer/trainer.yml')
            cascadePath = "Cascades/haarcascade_frontalface_default.xml"
            faceCascade = cv2.CascadeClassifier(cascadePath);
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Define min window size to be recognized as a face
            minW = 64
            minH = 48
            id = 0
            names = ['unknown', 'name1', 'name2', 'name3']
            gray = cv2.cvtColor(cropped,cv2.COLOR_BGR2GRAY)
            cv2.imshow('gray',gray)

            faces = faceCascade.detectMultiScale( 
                gray,
                scaleFactor = 1.2,
                minNeighbors = 5,
                minSize = (int(minW), int(minH)),
               )

            for(x,y,w,h) in faces:

                 cv2.rectangle(cropped, (x,y), (x+w,y+h), (0,255,0), 2)

                 id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

                 if (confidence < 100):
                      id = names[id]
                      confidence = "  {0}%".format(round(100 - confidence))
                      print ("in point one")
                 else:
                      id = "unknown"
                      confidence = "  {0}%".format(round(100 - confidence))
                      print ("in point two")
 
                 cv2.putText(cropped, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
                 cv2.putText(cropped, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)
            

    cv2.imshow('video',frame)

    k = cv2.waitKey(30) & 0xff
    if k == 27: # press 'ESC' to quit
        break

cap.release()
cv2.destroyAllWindows()

