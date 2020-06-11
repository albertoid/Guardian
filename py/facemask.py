from PIL import Image
import os
from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm.notebook import tqdm
import cv2
import pickle
from sklearn.model_selection import train_test_split as tts
import pandas as pd
from time import sleep
from IPython.display import clear_output

xVectorSize=100
yVectorSize=100

def getImageVector(img):
    #img = Image.open(img)
    img = img.resize( (xVectorSize, yVectorSize) )
    img = img.convert('L')
    return imgToVector(img) 

#Funcion para vectorizar imagenes y asignales clasificación
def imgToVector(img,x=xVectorSize,y=yVectorSize,classification=1):
    vector=[]
    vector.append(classification)

    for j in range(y):
        for i in range(x):
            vector.append(img.getpixel((i,j)))
    return vector

def reconstructor(vector,x=xVectorSize,y=yVectorSize):
    reb=Image.new('L',(x,y))
    for j in range(y):
        for i in range(x):
            reb.putpixel((i,j),vector[i+j*x]) 
    return reb

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def faceslocation(img):
    original_image = cv2.imread(img)
    grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    detected_faces = face_cascade.detectMultiScale(grayscale_image)
    return detected_faces

def getfaces(img):
    
    source = Image.open(img)
    source = source.convert('L')
    
    detected_faces=faceslocation(img)

    
    faces=[]
    
    for i in detected_faces:
        tlbr=list(i)
        
        x=int(tlbr[2])
        y=int(tlbr[3])
        x_offset=int(tlbr[0])
        y_offset=int(tlbr[1])

        face=Image.new('L',(x,y))

        source
        for j in range(y):
            for i in range(x):
                pixel=source.getpixel((i+x_offset,j+y_offset))
                face.putpixel((i,j),pixel) 
        faces.append(face)

    return faces

def textInImage(img,text='Prueba Texto'):
    font                   = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (50,50)
    fontScale              = 1
    fontColor              = (255,255,255)
    lineType               = 2

    cv2.putText(img,text, 
        bottomLeftCornerOfText, 
        font, 
        fontScale,
        fontColor,
        lineType)

xgbc = []
with (open('xgbc100x100', 'rb')) as openfile:
    while True:
        try:
            xgbc.append(pickle.load(openfile))
        except EOFError:
            break
xgbc=xgbc[0]

cam = cv2.VideoCapture(0)
cv2.startWindowThread()

cv2.namedWindow("test")

stop=False
while stop==False:
    ret, frame = cam.read()
        
    
    frame50 = rescale_frame(frame,percent=50)
    cv2.imshow("test", frame50)
    img='current_img.png'
    
    cv2.imwrite(img, frame)
    faces=getfaces(img)
    print(chr(27) + "[2J") #clear_output(wait=True)
    
    for i in range(len(faces)):
        
        X=pd.DataFrame(getImageVector(faces[i])).T
        X=X.drop(0,axis=1)
        prediccion=xgbc.predict(X)
        X=list(X.T[0])
        textInImage(frame50,'Sin Tapabocas') if prediccion == 0 else textInImage(frame50,'Con Tapabocas')
        print('💀',i) if prediccion == 0 else print('😷',i)
        
    cv2.imshow("test", frame50)
     
    k = cv2.waitKey(1)
    if k%256 == 27: # ESC pressed
        print("Escape hit, closing...")
        stop=True
    


cam.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
print('End')