import numpy as np
import cv2
import os
from imutils.object_detection import non_max_suppression
from PIL import Image
from statistics import mean
import math  
from playsound import playsound
import pickle

def calc_dist_x(array, el_array):
    dist_l = []
    if isinstance(array, (np.ndarray, np.generic) ) == True:
        for i in array:
            h1 = abs(i[1]-i[3])
            h2 = abs(el_array[1]-el_array[3])
            h_ref = max(h1,h2)
            
            dH1 = i[0]-el_array[0]
            dH2 = i[2]-el_array[2]
            dH_av = abs(mean([dH1,dH2]))
            dH_av = dH_av/(2*h_ref)
            
            #print("h_ref: ",h_ref)
            #print("dH_av: ",dH_av)
            if dH_av > 0:
                dist_l.append(dH_av)

        if len(dist_l)>0:
            #print(dist_l)
            return min(dist_l)
        else:
            return 3
    else:
        return 3


# Cargar el pickle
xgbc = []
with (open("xgbc100x100", "rb")) as openfile:
    while True:
        try:
            xgbc.append(pickle.load(openfile))
        except EOFError:
            break
xgbc=xgbc[0]


# initialize the HOG descriptor/person detector
body_det = cv2.HOGDescriptor()
body_det.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

face_det = cv2.CascadeClassifier(os.path.join('./models/haarcascade_frontalface_default.xml'))

cv2.startWindowThread()

# open webcam video stream
cap = cv2.VideoCapture(0)

# the output will be written to output.avi
out = cv2.VideoWriter(
    'output.avi',
    cv2.VideoWriter_fourcc(*'MJPG'),
    15.,
    (640,480))

label_d = {0:"No_Mask",1:"Mask"}
color_d = {0: (0,0,255), 1: (255,0,0)}

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # resizing for faster detection
    frame = cv2.resize(frame, (640, 480))
    # flipping image to mirror
    frame =cv2.flip(frame,1,1) 
    
    # using a greyscale picture, also for faster detection
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # detect people in the image
    # returns the bounding boxes for the detected people
    # TRY WITH GRAY AND FRAME
    body_boxes, weights = body_det.detectMultiScale(gray, winStride=(8,8) )
    body_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in body_boxes])
    
    # detect faces in the image
    # returns the bounding boxes for the detected faces
    face_boxes = face_det.detectMultiScale(gray)
    face_boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in face_boxes])
    
    # apply non-maxima suppression to the bounding boxes using overlap threshold to try to maintain overlapping
    # boxes that are still identificable
    body_pick = non_max_suppression(body_boxes, probs=None, overlapThresh=0.5)
    face_pick = non_max_suppression(face_boxes, probs=None, overlapThresh=0.5)
    
    # Testing Results
    #print("body_pick: ",body_pick)
    #print("type: ",type(body_pick))
    #print("dimentions: ", body_pick.shape)
    
    # Checking number of people in frame
    #num_p = num_of_people(face_pick)
    
    
    
    for element in body_pick:
        (xA, yA, xB, yB) = element
        min_dist = calc_dist_x(body_pick, element)
        print("min_dist: ",min_dist)
        if (min_dist > 0) & (min_dist < .2):
            color = 0
        else:
            color = 1
        
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB), color_d[color], 2)
        cv2.putText(frame, "human", (xA, yA-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        
        #print("points: {},{},{},{}".format(xA, yA, xB, yB))
        
    for element in face_pick:
        (xA, yA, xB, yB) = element
        
        # Saving FACE IMAGE to a VAR
        face_img = gray[yA:yB, xA:xB]
        resized=cv2.resize(face_img,(100,100))
        #cv2.imwrite("face.jpg", resized)
        #pred = xgbc.predict(resized)
        #print("prediction: ",pred)
        
        # display the detected boxes in the colour picture
        cv2.rectangle(frame, (xA, yA), (xB, yB), (255,255,255), 2)
        cv2.putText(frame, "face", (xA, yA-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2)
        
        
        
    # Write the output video 
    out.write(frame.astype('uint8'))
    # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
# and release the output
out.release()
# finally, close the window
for i in range(1,10):
    cv2.destroyAllWindows()
    cv2.waitKey(1)