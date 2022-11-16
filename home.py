from fastapi import FastAPI,Request
from fastapi.responses import HTMLResponse
from starlette.responses import StreamingResponse
from fastapi.templating import Jinja2Templates

import uvicorn
import cv2
import os
import face_recognition
import imutils


app=FastAPI()
templates=Jinja2Templates(directory="templates")

cascPathface = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(cv2.__file__) + "/data/haarcascade_eye_tree_eyeglasses.xml"
faceCascade=cv2.CascadeClassifier(cascPathface)
eyeCascade=cv2.CascadeClassifier(cascPatheyes)

capture=cv2.VideoCapture(-0)
capture.set(3,640)
capture.set(3,480)
   
def gen_frames():
    while True:
        ret,img=capture.read()
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2BGRA)
        faces=faceCascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(60,60),flags=cv2.CASCADE_SCALE_IMAGE)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            faceROI=img[y:y+h,x:x+w]
            eyes=eyeCascade.detectMultiScale(faceROI)
            for(x2,y2,w2,h2) in eyes:
                eye_centre=(x+x2+w2//2,y+y2+h2//2)
                radius=int(round((w2+h2)*0.25))
                frame=cv2.circle(img,eye_centre,radius,(255,0,0),4)
        streaming_frame=cv2.imencode('.jpg',img)[1].tobytes()
        yield(b'--streaming_frame\r\n' b'Content-Type: image/jpeg\r\n\r\n'+bytearray(streaming_frame) + b'\r\n')
        key=cv2.waitKey(1)
        if key==27:
            break   


@app.get("/")
def home_page(request=Request,response_class=HTMLResponse):
    return  templates.TemplateResponse('index.html',context={"request":request})

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(), media_type="multipart/x-mixed-replace;boundary=streaming_frame")
