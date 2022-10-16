import cv2 as cv
import mediapipe as mp
import time
cTime=0
pTime=0

cap=cv.VideoCapture(0,cv.CAP_DSHOW)
#cap=cv.VideoCapture(r"videos\8.mp4")
mpHands=mp.solutions.hands
hands=mpHands.Hands()#Dentro da biblioteca: .Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
mpDraw=mp.solutions.drawing_utils#Comando para desenhar linhas e pontos na mão
"""DENTRO DA BIBLIOTECA
static_image_mode
    static_image_mode=False=> A função rastrea e detecta, dependendo da confiabilidade(%) do rastreamento na imagem.
        Se tiver uma boa confiabilidade no rastreamento, a função não faz a detecção novamente.
        Se a confiabilidade do rastreamento estiver ruim, a função faz a detecção novamente.
    static_image_mode=True=> Vai o tempo todo fazer a detecção, ou seja, deixará o programa mais lento. 
max_num_hands=> Número máximo de mãos na figura.
min_detection_confidence=> Mínima de confiança na detecção.
min_tracking_confidence=> Mínima de confiança no rastreamento."""


while True:
    _,frame=cap.read()
    frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results=hands.process(frameRGB)
    #print(results.multi_hand_landmarks)#Fala se algo foi ou não rastreado na câmera
    if(results.multi_hand_landmarks):#Se algo for detectado...
        for handLms in results.multi_hand_landmarks:#Pega os landmarks de cada mão detectada dentro de "results.multi_hand_landmarks"
            mpDraw.draw_landmarks(frame,handLms,mpHands.HAND_CONNECTIONS)#Desenha cada landmark e cada linha entre os landsmarks em cada mão detectada
            for id,lm in enumerate(handLms.landmark):#Pega o id e as coordenadas(x,y,z) de cada landmark que forma a mão
                #print(f"{id}\n{lm}")
                h,w,c=frame.shape#Pega a altura(h), largura(w) e os canais(c) dos frames
                cx,cy=int(lm.x*w),int(lm.y*h)#Cordenada x e y dos landmarks em relação ao frame inteiro
                print(id, cx,cy)
                cv.circle(frame,(cx,cy),5,(0,255,0),cv.FILLED)
                if id==4:
                    cv.circle(frame,(cx,cy),5,(255,0,255),cv.FILLED)
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    cv.putText(frame,str(int(fps)),(10,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(R,G,B),espessura)
    cv.imshow("Cam",frame)
    key=cv.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
cv.destroyAllWindows()
