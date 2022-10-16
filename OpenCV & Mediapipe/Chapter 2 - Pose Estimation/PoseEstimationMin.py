import cv2 as cv
import mediapipe as mp
import time
cTime=0
pTime=0
mpDraw=mp.solutions.drawing_utils
mpPose=mp.solutions.pose
pose=mpPose.Pose()
"""DENTRO DA BIBLIOTECA
static_image_mode
    static_image_mode=False=> A função rastrea e detecta, dependendo da confiabilidade(%) do rastreamento na imagem.
        Se tiver uma boa confiabilidade no rastreamento, a função não faz a detecção novamente.
        Se a confiabilidade do rastreamento estiver ruim, a função faz a detecção novamente.
    static_image_mode=True=> Vai o tempo todo fazer a detecção, ou seja, deixará o programa mais lento. 
upper_body_only=> True
    Somente pega detecta a parte de cima do corpo.
upper_body_only=> False
    Detecta todo o corpo.
smooth_landmarks=>???????????????????????????????????????????????????????????
min_detection_confidence=> Mínima de confiança na detecção.
min_tracking_confidence=> Mínima de confiança no rastreamento.
"""

cap=cv.VideoCapture(r"videos\8.mp4")
#cap=cv.VideoCapture(0,cv.CAP_DSHOW)

while True:
    _,frame=cap.read()
    frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results=pose.process(frameRGB)
    #print(results.pose_landmarks)#Fala se algo foi ou não rastreado na câmera ou vídeo
    if(results.pose_landmarks):#Se algo for detectado...
        mpDraw.draw_landmarks(frame,results.pose_landmarks,mpPose.POSE_CONNECTIONS)#Desenha os pontos(landmarks) e as linhas entre eles
        for id, lm in enumerate(results.pose_landmarks.landmark):#Pega os landmarks de cada mão detectada dentro de "results.pose_landmarks"
            h,w,c=frame.shape
            print(f"{id}\n{lm}")
            cx,cy=int(lm.x*w),int(lm.y*h)
            cv.circle(frame,(cx,cy),3,(255,0,0),cv.FILLED)
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv.putText(frame,f"FPS: {(fps)}",(5,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
    key=cv.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
    cv.imshow("Video",frame)