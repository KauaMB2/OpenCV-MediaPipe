import numpy as np
import cv2 as cv
import os
import HandtrackingModule as hm
import time
import copy


brushThickness=15
eraserThickness=50
xp,yp=0,0
imgCanvas=np.zeros((480,640,3),np.uint8)
pTime=0
cTime=0
folderPath=r"imgs/PaintImages"
myList=os.listdir(folderPath)
print(myList)
overlaylist=[]
for imPath in myList:
    image=cv.imread(f"{folderPath}/{imPath}")
    overlaylist.append(image)
print(len(overlaylist))
header=overlaylist[0]
drawColor=(0,0,255)
cap=cv.VideoCapture(0,cv.CAP_DSHOW)
wCam, hCam = 640, 480
cap.set(3, wCam)
cap.set(4, hCam)

detector=hm.handDetector()
while True:
    #1. Import image
    _,frame=cap.read()
    frame=cv.flip(frame,1)#Vira a imagem horizontalmente para tirar o espelhamento
    
    #2. Find hand landmarks
    debug_frame = copy.deepcopy(frame)#Pega o "frame" e joga no debug_frame
    frame=detector.findHands(frame,debug_frame,False,True)
    lmList=detector.findPosition(frame,draw=False)
    if len(lmList) != 0:
        #print(lmList)

        #tip os index and middle fingers(Pontas do dedo indicador e do meio)
        x1,y1=lmList[8][1:]
        x2,y2=lmList[12][1:]

        #3. Check wich fingers are up
        fingers=detector.fingersUp()
        #print(fingers)

        #4. If selection mode - Two fingers are up
        if fingers[1] and fingers[2]:#Se ativar o modo selecionar
            xp,yp=0,0#Zera as posições do desenho para não bugar o desenho
            print("Selection Mode")
            if y1<79:
                if 115<x1<215:
                    header=overlaylist[0]
                    drawColor=(0,0,255)
                elif 215<x1<315:
                    header=overlaylist[1]
                    drawColor=(0,255,0)
                elif 315<x1<415:
                    header=overlaylist[2]
                    drawColor=(255,0,0)
                elif 415<x1<545:
                    header=overlaylist[3]
                    drawColor=(0,0,0)
            cv.rectangle(frame,(x1,y1-25),(x2,y2+25),drawColor,cv.FILLED)
        #5. If Drawing - Index(Indicador) finger is up
        if fingers[1] and fingers[2]==False:
            print("Drawing Mode")
            cv.circle(frame,(x1,y1),15,drawColor,cv.FILLED)
            if xp==0 and yp==0:
                xp,yp=x1,y1
            if drawColor==(0,0,0):
                cv.line(frame,(xp,yp),(x1,y1),drawColor,eraserThickness)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,eraserThickness)
            else:
                cv.line(frame,(xp,yp),(x1,y1),drawColor,brushThickness)
                cv.line(imgCanvas,(xp,yp),(x1,y1),drawColor,brushThickness)
            xp,yp=x1,y1
        if fingers[1]==False:#Se o dedo indicador abaixar
            xp,yp=0,0#Zera as posições do desenho para não bugar o desenho
    else:#Se não achou nenhuma mão...
        xp,yp=0,0#Zera as posições do desenho para não bugar o desenho
    imgGray=cv.cvtColor(imgCanvas,cv.COLOR_BGR2GRAY)#Transforma "imgCanvas" em uma imagem cinza
    _,imgInv=cv.threshold(imgGray,0,255,cv.THRESH_BINARY_INV)#Transforma "imgCanvas" em uma imagem binária(preto ou branco)
    imgInv=cv.cvtColor(imgInv,cv.COLOR_GRAY2BGR)#Transforma "imgInv" em uma imagem binária(preto ou branco)
    frame=cv.bitwise_and(frame,imgInv)#Faz operação "And" entre imagem "frame" e "imgInv"
    frame=cv.bitwise_or(frame,imgCanvas)#Faz operação "Ou" entre imagem "frame" e "imgCanvas"
    
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv.putText(frame,f"FPS: {fps}",(25,120),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(R,G,B),espessura)
    frame[0:79,0:640]=header
    #frame=cv.addWeighted(frame,0.5,imgCanvas,0.5,0)
    cv.imshow("Cam",frame)
    #cv.imshow("imgCanvas",imgCanvas)
    #cv.imshow("imgInv",imgInv)
    key=cv.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
cv.destroyAllWindows()