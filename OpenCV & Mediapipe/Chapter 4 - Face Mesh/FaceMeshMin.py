import cv2 as cv
import mediapipe as mp
import time

#cap=cv.VideoCapture(r"videos\9.mp4")
cap=cv.VideoCapture(0,cv.CAP_DSHOW)
cTime=0
pTime=0
RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)
mpDraw=mp.solutions.drawing_utils
mpFaceMesh=mp.solutions.face_mesh
faceMesh=mpFaceMesh.FaceMesh(max_num_faces=2)
drawSpec=mpDraw.DrawingSpec(thickness=1,circle_radius=2,color=GREEN_COLOR)

while True:
    _,frame=cap.read()
    frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results=faceMesh.process(frameRGB)
    if (results.multi_face_landmarks):
        for faceLms in results.multi_face_landmarks:
            mpDraw.draw_landmarks(frame,faceLms,mpFaceMesh.FACEMESH_TESSELATION,drawSpec,drawSpec)
            h,w,c=frame.shape
            points=faceLms.landmark
            for id,point in enumerate(points):#Pega todos os pontos no rosto
                cx=int(point.x*w)
                cy=int(point.y*h)
                print(f"{id}\n{point}")
                cv.circle(frame,(cx,cy),3,RED_COLOR,cv.FILLED)#Desenha todos os pontos no rosto
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv.putText(frame,f"FPS: {(fps)}",(5,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
    key=cv.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
    cv.imshow("Video",frame)
