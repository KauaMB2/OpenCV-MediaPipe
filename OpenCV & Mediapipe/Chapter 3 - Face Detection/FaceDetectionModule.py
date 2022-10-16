import cv2 as cv
import mediapipe as mp
import time

class FaceDetector():
    def __init__(self,minDetectionConfidence=0.5):
        self.minDetectionConfidence=minDetectionConfidence
        self.mpFaceDetection=mp.solutions.face_detection
        self.mpDraw=mp.solutions.drawing_utils
        self.faceDetection=self.mpFaceDetection.FaceDetection(min_detection_confidence=self.minDetectionConfidence)

    def findFaces(self,frame):#Encontra e exibe a face
        frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.results=self.faceDetection.process(frameRGB)
        bboxs=[]
        if (self.results.detections):
            for id,detection in enumerate(self.results.detections):
                #print(id,detection)
                #print(detection.score)
                #print(detection.location_data.relative_keypoints)
                #print(detection.location_data.relative_bounding_box)
                #mpDraw.draw_detection(frame,detection)#Desenha a caixa delimitadora e todos os pontos no rosto
                bboxC=detection.location_data.relative_bounding_box#Pega as coordenadas do rosto dentro da "caixa delimitadora"
                h,w,c=frame.shape
                bbox=int(bboxC.xmin*w),int(bboxC.ymin*h),int(bboxC.width*w),int(bboxC.height*h)#Referencia os valores dentro da caixa delimitadora em relação ao tamanho do frame
                bboxs.append([bbox,detection.score])
                #print(f"{bboxC}\n{bbox}")
                cv.rectangle(frame, (bbox[0], bbox[1]-40), (bbox[0]+bbox[2], bbox[1]),(0, 0, 0), -1)#cor=(B,G,R)
                cv.putText(frame,f"{int(detection.score[0]*100)}%",(bbox[0]+5,bbox[1]-10),cv.FONT_HERSHEY_PLAIN,2,(0,255,0),4,cv.LINE_AA)
                cv.rectangle(frame,bbox,(255,0,255),2)#Desenha um retângulo ao redor do rosto
                self.fancyDraw(frame,bbox)#Chama função para desenhar as linhas mais grossas nas extremidades do bouding box
                points=detection.location_data.relative_keypoints#Pega a localização de todos os pontos do rosto
                for point in points:#Pega todos os pontos no rosto
                    cx=int(point.x*w)
                    cy=int(point.y*h)
                    cv.circle(frame,(cx,cy),3,(255,0,0),cv.FILLED)#Desenha todos os pontos no rosto
        return frame,bboxs
    def fancyDraw(self,frame,bbox,l=30,t=10):
        x,y,w,h=bbox
        x1,y1=x+w,y+h
        #DESENHA AS LINHAS MAIS GROSSAS NAS EXTREMIDADES DO BOUNDING BOX
        #Top Left x,y
        cv.line(frame,(x,y),(x+l,y),(255,0,255),t)
        cv.line(frame,(x,y),(x,y+l),(255,0,255),t)
        #Top Right x1,y
        cv.line(frame,(x1,y),(x1-l,y),(255,0,255),t)
        cv.line(frame,(x1,y),(x1,y+l),(255,0,255),t) 
        #Bottom Left x,y1
        cv.line(frame,(x,y1),(x+l,y1),(255,0,255),t)
        cv.line(frame,(x,y1),(x,y1-l),(255,0,255),t) 
        #Bottom Left x1,y1
        cv.line(frame,(x1,y1),(x1-l,y1),(255,0,255),t)
        cv.line(frame,(x1,y1),(x1,y1-l),(255,0,255),t) 
        
def main():
    #cap=cv.VideoCapture(r"videos\9.mp4")
    cap=cv.VideoCapture(0,cv.CAP_DSHOW)
    cTime=0
    pTime=0
    detector=FaceDetector()
    while True:
        _,frame=cap.read()
        frame,bboxs=detector.findFaces(frame)
        cTime=time.time()
        fps=int(1/(cTime-pTime))
        pTime=cTime
        cv.putText(frame,f"FPS: {(fps)}",(5,70),cv.FONT_HERSHEY_PLAIN,3,(255,0,255),3)#putText(frame,text,(positionX,positionY),font,tamanho,(B,G,R),espessura)
        key=cv.waitKey(1)#ESC = 27
        if key==27:#Se apertou o ESC
            break
        cv.imshow("Video",frame)
if __name__=="__main__":
    main()