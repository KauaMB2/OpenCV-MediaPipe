import cv2 as cv
import HandtrackingModule as hm
import numpy as np
import time
import autopy
import copy

cTime=0
pTime=0
cap=cv.VideoCapture(0,cv.CAP_DSHOW)
#cap=cv.VideoCapture(r"videos\8.mp4")
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)
frameR=100#Frame Reduction
smoothening=5
pLocX,pLocY=0,0
cLocX,cLocY=0,0
detector=hm.handDetector()
wScr,hScr=autopy.screen.size()
print(wScr,hScr)
while True:
    #1. Import image
    _,frame=cap.read()
    frame=cv.flip(frame,1)#Vira a imagem horizontalmente para tirar o espelhamento
    debug_frame = copy.deepcopy(frame)#Pega o "frame" e joga no debug_frame
    frame=detector.findHands(frame,debug_frame)
    lmList=detector.findPosition(frame,False)
    cv.rectangle(frame,(frameR,frameR),(wCam-frameR,hCam-frameR),(255,0,255),2)
    #2. Get the tip of the index(indicador) and middle fingers
    if len(lmList) != 0:
        x1,y1=lmList[8][1:]
        x2,y2=lmList[8][1:]
        #print(x1,y1,x2,y2)
        #3. Check which fingers are up
        fingers=detector.fingersUp()
        #print(fingers)
        #4. Only index Finger: Moving Mode
        if fingers[1]==True and fingers[2]==False:
            #5. Convert coordinates
            x3=np.interp(x1,(frameR,wCam-frameR),(0,wScr))
            y3=np.interp(y1,(frameR,hCam-frameR),(0,hScr))
            cv.circle(frame,(x1,y1),15,(255,0,255),cv.FILLED)
            #6. Smoothen Values
            cLockX=pLocX+(x3-pLocX)/smoothening
            cLocky=pLocY+(x3-pLocY)/smoothening
            #7. Move mouse
            autopy.mouse.move(x3,y3)
            pLocX,pLocY=cLocX,cLocY
        #8. Both index and middle fingers are up: Clicking Mode
        if fingers[1]==True and fingers[2]==True:
            #9. Find distance between fingers
            length,frame,lineInfo=detector.findDistance(8,12,frame)
            print(length)
            #10. Click moyse if distance short
            if length<40:
                cv.circle(frame,(lineInfo[4],lineInfo[5]),15,(0,255,0),cv.FILLED)
                autopy.mouse.click()
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv.putText(frame,f"FPS: {fps}",(25,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(R,G,B),espessura)
    cv.imshow("Cam",frame)
    key=cv.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
cv.destroyAllWindows()