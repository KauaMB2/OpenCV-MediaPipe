import cv2 as cv
import mediapipe as mp
import time

#cap=cv.VideoCapture(r"videos\9.mp4")
cap=cv.VideoCapture(0,cv.CAP_DSHOW)
cTime=0
pTime=0
mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection()

while True:
    _,frame=cap.read()
    frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    results=faceDetection.process(frameRGB)
    if (results.detections):
        for id,detection in enumerate(results.detections):
            #print(id,detection)
            #print(detection.score)
            #print(detection.location_data.relative_keypoints)
            #print(detection.location_data.relative_bounding_box)
            #mpDraw.draw_detection(frame,detection)#Desenha a caixa delimitadora e todos os pontos no rosto
            bboxC=detection.location_data.relative_bounding_box#Pega as coordenadas do rosto dentro da "caixa delimitadora"
            h,w,c=frame.shape
            bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)#Referencia os valores dentro da caixa delimitadora em relação ao tamanho do frame
            #print(f"{bboxC}\n{bbox}")
            cv.rectangle(frame, (bbox[0], bbox[1]-40), (bbox[0]+bbox[2], bbox[1]),(0, 0, 0), -1)#cor=(B,G,R)
            cv.putText(frame,f"{int(detection.score[0]*100)}%",(bbox[0]+5,bbox[1]-10),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),4,cv.LINE_AA)
            cv.rectangle(frame,bbox,(255,0,255),5)#Desenha um retângulo ao redor do rosto
            points=detection.location_data.relative_keypoints#Pega a localização de todos os pontos do rosto
            for point in points:#Pega todos os pontos no rosto
                cx=int(point.x*w)
                cy=int(point.y*h)
                cv.circle(frame,(cx,cy),3,(255,0,0),cv.FILLED)#Desenha todos os pontos no rosto
    cTime=time.time()
    fps=int(1/(cTime-pTime))
    pTime=cTime
    cv.putText(frame,f"FPS: {(fps)}",(5,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
    key=cv.waitKey(1)#ESC = 27
    if key==27:#Se apertou o ESC
        break
    cv.imshow("Video",frame)