import cv2 as cv
import time
import os
import HandtrackingModule as hm
import time
import copy
"""
                    ATENÇÃO
    O CÓDIGO SÓ FUNCIONA PARA A MÃO DIREITA!!!!
"""
wCam, hCam = 640, 480
cap = cap=cv.VideoCapture(0,cv.CAP_DSHOW)
cap.set(3, wCam)
cap.set(4, hCam)
folderPath = r"imgs/FingerImages"
myList = os.listdir(folderPath)
print(myList)
overImageList = []
for imPath in myList:
    image=cv.imread(f"{folderPath}/{imPath}")
    overImageList.append(image)

cTime=0
pTime = 0
detector = hm.handDetector(maxHands=1)
tipIds = [4, 8, 12, 16, 20]
while True:
    _, frame = cap.read()
    h,w,c=overImageList[0].shape
    debug_frame = copy.deepcopy(frame)#Pega o "frame" e joga no debug_frame
    frame = detector.findHands(frame,debug_frame)
    lmList = detector.findPosition(frame, draw=False)
    #print(lmList)
    if len(lmList) !=0:
        fingers=[]

        #Dedão - Verifica se o de dedão está ou não fechado.
        if lmList[tipIds[0]][1] > lmList[tipIds[0]-1][1]:#Se sim...
            fingers.append(True)#Adiciona "True" dentro da lista, indicando que a ponta do dedão ESTÁ encolhida(Dedão fechado) 
        else:#Se não...
            fingers.append(False)#Adiciona "False" dentro da lista, indicando que a ponta dedão NÃO ESTÁ encolhida(Dedão aberto)
        #Outros 4 dedos
        for id in range(1,5):#Verifica se a ponta decada dedo está ou não abaixo do meio do dedo 
            if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:#Se sim...
                fingers.append(True)#Adiciona "True" dentro da lista, indicando que a ponta do dedo está ABAIXO do meio do dedo, então o dedo ESTÁ dobrado
            else:#Se não...
                fingers.append(False)#Adiciona "False" dentro da lista, indicando que a ponta do dedo está ACIMA do meio do dedo, então o dedo NÃO ESTÁ dobrado
        #print(fingers)
        totalFingers=fingers.count(True)
        print(totalFingers)
        h,w,c=overImageList[totalFingers].shape
        frame[0:h,0:w]=overImageList[totalFingers]
        cv.putText(frame,str(totalFingers),(30,330),cv.FONT_HERSHEY_PLAIN,6,(255,0,255),6)#putText(frame,text,(positionX,positionY),font,tamanho,(R,G,B),espessura)
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv.putText(frame,f"FPS: {fps}",(470,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(R,G,B),espessura)
    cv.imshow("Cam",frame)
    key=cv.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
cv.destroyAllWindows()