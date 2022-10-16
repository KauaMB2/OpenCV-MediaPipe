import cv2 as cv
import mediapipe as mp
import numpy as np
import time
import HandtrackingModule as hm
import copy
from ctypes import cast,POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
import math

devices=AudioUtilities.GetSpeakers()
interface=devices.Activate(IAudioEndpointVolume._iid_,CLSCTX_ALL,None)
volume=cast(interface,POINTER(IAudioEndpointVolume))
volRange=volume.GetVolumeRange()
minVol=volRange[0]
maxVol=volRange[1]
print(volRange)
vol=0
volBar=volume.GetMasterVolumeLevel()
pTime=0
cTime=0
cap=cv.VideoCapture(0,cv.CAP_DSHOW)
###################
wCam,hCam=648,488
###################
cap.set(3,wCam)#Define a largura do frame
cap.set(4,hCam)#Define a altura do frame
detector=hm.handDetector()
while True:
    _,frame=cap.read()
    debug_frame = copy.deepcopy(frame)#Pega o "frame" e joga no debug_frame
    frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    frame=detector.findHands(frame,debug_frame)
    lmList=detector.findPosition(frame,draw=False)
    if len(lmList)!=0:
        x1,y1=lmList[4][1],lmList[4][2]
        x2,y2=lmList[8][1],lmList[8][2]
        #print(lmList)
        middleX,middleY=int((x1+x2)/2),int((y1+y2)/2)
        cv.circle(frame,(x1,y1),15,(255,0,255),cv.FILLED)
        cv.circle(frame,(x2,y2),15,(255,0,255),cv.FILLED)
        cv.line(frame,(x1,y1),(x2,y2),(255,0,255),3)
        cv.circle(frame,(middleX,middleY),15,(255,0,255),cv.FILLED)
        #Hand range 50 - 300
        #Volume range -95.0 - 0
        length=math.hypot(x2-x1,y2-y1)#Calcula o tamanho da reta entre os dedos
        vol=np.interp(length,[50,250],[minVol,maxVol])
        volBar=np.interp(length,[50,300],[400,150])
        volPer = np.interp(length, [50, 300], [0, 100])
        #print(int(volBar),vol)
        volume.SetMasterVolumeLevel(vol,None)
        if length < 50:
            cv.circle(frame, (middleX, middleY), 15, (0, 255, 0), cv.FILLED)
    else:#Se não tiver nenhuma mão na tela...
        volBar=volume.GetMasterVolumeLevel()
        try:
            volBar=(1/(float(volBar)*(-1)))*100
            if(volBar>90):
                volBar=100
        except:
            volBar=100
        #print(volBar)
        volPer=math.log(volBar,(1.047)) #Passa da escala logarítmica para uma escala de porcentagem do som(%)
                                        #Cálculo aproximado. A fórmula eu encontrei através de cálculos com função exponencial
        volBar=np.interp(volBar,[0,100],[400,150])
    cv.rectangle(frame,(50,150),(85,400),(0,255,0),3)
    cv.rectangle(frame,(50,int(volBar)),(85,400),(0,255,0),cv.FILLED)
    cv.putText(frame, f'{int(volPer)} %', (40, 450), cv.FONT_HERSHEY_COMPLEX,1, (255, 0, 0), 3)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(R,G,B),espessura)
    cv.imshow("Cam",frame)
    key=cv.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
cv.destroyAllWindows()