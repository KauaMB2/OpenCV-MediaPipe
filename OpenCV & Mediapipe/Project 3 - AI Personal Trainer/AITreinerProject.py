import cv2 as cv
import numpy as np
import time
import copy
import numpy as np
import PoseEstimationModule as pm

cTime=0
pTime=0
count=0
dir=0

#cap=cv.VideoCapture(r"videos\12.mp4")
cap=cv.VideoCapture(0,cv.CAP_DSHOW)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
detector=pm.poseDetector()
while True:
    _,frame=cap.read()
    #frame=cv.imread(r"imgs/ExercicesImages/0.jpeg")
    frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    debug_frame = copy.deepcopy(frame)#Pega o "frame" e joga no debug_frame
    frame=detector.findPose(frame,debug_frame,False,False)
    lmList=detector.findPosition(frame,False)
    if len(lmList) != 0:
        #Right Arm
        angleR=detector.findAngle(frame,12,14,16)
        perR=np.interp(angleR,(200,340),(0,100))
        perR=int(perR)#Tira casas decimais
        barR=np.interp(angleR,(210,310),(420,80))
        barR=int(barR)#Tira casas decimais
        if perR==100:
            if dir==0:
                count+=0.5
                dir=1
        if perR==0:
            if dir==1:
                count+=0.5
                dir=0
        print(perR,count)
       
        #Desenha barra
        if(perR<=30):
            cor=(255,0,255)
        else:
            cor=(0,255,0)
        cv.rectangle(frame,(550,80),(600,420),cor,3)
        cv.rectangle(frame,(550,int(barR)),(600,420),cor,cv.FILLED)
        cv.putText(frame,f"{perR}%",(525,75),cv.FONT_HERSHEY_PLAIN,3,cor,5)
        
        #Contador de supinos
        cv.rectangle(frame,(0,380),(160,480),(0,255,0),cv.FILLED)
        cv.putText(frame,str(count),(10,450),cv.FONT_HERSHEY_PLAIN,5,(255,0,0),5)
    
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    
    #Verifica se os halteres deram curvas
    cv.putText(frame,f"FPS: {(fps)}",(5,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
    key=cv.waitKey(1)#ESC = 27
    
    if key==27:#Se apertou o ESC
        break
    cv.imshow("Video",frame)