import cv2 as cv
import mediapipe as mp
import time
import copy
import numpy as np
import math
def calc_bounding_rect(frame, landmarks):#Função para calcular a borda do retângulo
    frame_width, frame_height = frame.shape[1], frame.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * frame_width), frame_width - 1)
        landmark_y = min(int(landmark.y * frame_height), frame_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv.boundingRect(landmark_array)
    return [x, y, x + w, y + h]

class handDetector():
    def __init__(self,mode=False,maxHands=2,detectionConfidence=0.5,trackConfidence=0.5):
        self.mode=mode
        self.maxHands=maxHands
        self.detectionConfidence=detectionConfidence
        self.trackConfidence=trackConfidence
        self.mpHands=mp.solutions.hands
        self.hands=self.mpHands.Hands(static_image_mode=self.mode,max_num_hands=self.maxHands,min_detection_confidence=self.detectionConfidence,min_tracking_confidence=self.trackConfidence)#Dentro da biblioteca: .Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=0.5,min_tracking_confidence=0.5)
        """DENTRO DA BIBLIOTECA
        static_image_mode
            static_image_mode=False=> A função rastrea e detecta, dependendo da confiabilidade(%) do rastreamento na imagem.
                Se tiver uma boa confiabilidade no rastreamento, a função não faz a detecção novamente.
                Se a confiabilidade do rastreamento estiver ruim, a função faz a detecção novamente.
            static_image_mode=True=> Vai o tempo todo fazer a detecção, ou seja, deixará o programa mais lento. 
        max_num_hands=> Número máximo de mãos na figura.
        min_detection_confidence=> Mínima de confiança na detecção.
        min_tracking_confidence=> Mínima de confiança no rastreamento."""
        self.mpDraw=mp.solutions.drawing_utils#Comando para desenhar linhas e pontos na mão
        self.tipIds=[4,8,12,16,20]

    def findHands(self,frame,debug_frame,drawBourding=True,drawHand=True):#Exibe mão e borda
        self.results=self.hands.process(frame)
        #print(self.results)#Fala se algo foi ou não rastreado na câmera
        if(self.results.multi_hand_landmarks):#Se algo for detectado...
            for handLms in self.results.multi_hand_landmarks:#Pega os landmarks de cada mão detectada dentro de "results.multi_hand_landmarks"
                if(drawHand):
                    self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)#Desenha cada landmark e cada linha entre os landsmarks em cada mão detectada
                if(drawBourding):
                    brect = calc_bounding_rect(debug_frame, handLms)
                    cv.rectangle(frame, (brect[0]-15, brect[1]-45), (brect[2]+10, brect[1]-15),(0, 0, 0), -1)
                    cv.putText(frame, "Hand", (brect[0]-10, brect[1] - 19),cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 4, cv.LINE_AA)
                    cv.rectangle(frame, (brect[0]-15, brect[1]-15), (brect[2]+10, brect[3]+10),(0, 255, 0), 5)
        return frame
        
    def findPosition(self,frame,draw=True):#Pega todas os landmarks e coloca dentro de uma lista "lmList" para customizar os landmarks e as ligações entre eles
        self.lmList=[]
        if(self.results.multi_hand_landmarks):#Se algo for detectado...
            for handLms in self.results.multi_hand_landmarks:#Pega os landmarks de cada mão detectada dentro de "results.multi_hand_landmarks"
                self.mpDraw.draw_landmarks(frame,handLms,self.mpHands.HAND_CONNECTIONS)#Desenha cada landmark e cada linha entre os landsmarks em cada mão detectada
                for id,lm in enumerate(handLms.landmark):#Pega o id e as coordenadas(x,y,z) de cada landmark que forma a mão
                    #print(f"{id}\n{lm}")
                    h,w,c=frame.shape#Pega a altura(h), largura(w) e os canais(c) dos frames
                    cx,cy=int(lm.x*w),int(lm.y*h)#Cordenada x e y dos landmarks em relação ao frame inteiro
                    #print(id, cx,cy)
                    self.lmList.append([id,cx,cy])
                    cv.circle(frame,(cx,cy),5,(0,0,255),cv.FILLED)#cor=(B,G,R)
                    if draw:
                        if id==4:
                            cv.circle(frame,(cx,cy),5,(255,0,255),cv.FILLED)#cor=(B,G,R)
        return self.lmList
    def fingersUp(self):#Verifica quais dedos estão para cima ou quais estão abaixados
        fingers=[]
        #Dedão - Verifica se o de dedão está ou não fechado.
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:#Se sim...
            fingers.append(True)#Adiciona "True" dentro da lista, indicando que a ponta do dedão ESTÁ encolhida(Dedão fechado) 
        else:#Se não...
            fingers.append(False)#Adiciona "False" dentro da lista, indicando que a ponta dedão NÃO ESTÁ encolhida(Dedão aberto)
        #Outros 4 dedos
        for id in range(1,5):#Verifica se a ponta decada dedo está ou não abaixo do meio do dedo 
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:#Se sim...
                fingers.append(True)#Adiciona "True" dentro da lista, indicando que a ponta do dedo está ABAIXO do meio do dedo, então o dedo ESTÁ dobrado
            else:#Se não...
                fingers.append(False)#Adiciona "False" dentro da lista, indicando que a ponta do dedo está ACIMA do meio do dedo, então o dedo NÃO ESTÁ dobrado
        #print(fingers)
        return fingers
    def findDistance(self, p1, p2, img, draw=True,r=15, t=3):#Verifica a distância entre dois dedos
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv.circle(img, (x1, y1), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (x2, y2), r, (255, 0, 255), cv.FILLED)
            cv.circle(img, (cx, cy), r, (0, 0, 255), cv.FILLED)
            length = math.hypot(x2 - x1, y2 - y1)

        return length, img, [x1, y1, x2, y2, cx, cy]
        
def main():
    cTime=0
    pTime=0
    cap=cv.VideoCapture(0,cv.CAP_DSHOW)
    #cap=cv.VideoCapture(r"videos\8.mp4")
    detector=handDetector() 
    while True:
        _,frame=cap.read()
        debug_frame = copy.deepcopy(frame)#Pega o "frame" e joga no debug_frame
        #ou...
        #debug_frame=frame
        frame=detector.findHands(frame,debug_frame)
        lmList=detector.findPosition(frame)
        #if len(lmList)!=0:
            #detector.findDistance(8,12,frame)
        #print(lmList)#Mostra as coordenadas dos landmarks
        cTime=time.time()
        fps=int(1/(cTime-pTime))
        pTime=cTime
        cv.putText(frame,f"FPS: {(fps)}",(5,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
        cv.imshow("Cam",frame)
        key=cv.waitKey(1)#ESC = 27
        if key==27:#Se apertou o ESC
            break
if __name__ == "__main__":
    main()
cv.destroyAllWindows()