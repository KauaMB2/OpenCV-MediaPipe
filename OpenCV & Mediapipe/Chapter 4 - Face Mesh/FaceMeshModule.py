import cv2 as cv
import mediapipe as mp
import time


RED_COLOR = (0, 0, 255)
GREEN_COLOR = (0, 128, 0)

class FaceMeshDetector():
    def __init__(self,staticMode=False,maxFaces=2,minDetectionConfidence=0.5,minTrackConfidence=0.5):
        self.staticMode=staticMode
        self.maxFaces=maxFaces
        self.minDetectionConfidence=minDetectionConfidence
        self.minTrackConfidence=minTrackConfidence
        self.mpDraw=mp.solutions.drawing_utils
        self.mpFaceMesh=mp.solutions.face_mesh
        self.faceMesh=self.mpFaceMesh.FaceMesh(static_image_mode=self.staticMode,max_num_faces=self.maxFaces,min_detection_confidence=self.minDetectionConfidence,min_tracking_confidence=self.minTrackConfidence)
        self.drawSpec=self.mpDraw.DrawingSpec(thickness=1,circle_radius=2,color=GREEN_COLOR)
    
    def findFaceMesh(self,frame):
        frameRGB=cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.results=self.faceMesh.process(frameRGB)
        faces=[]
        if (self.results.multi_face_landmarks):
            for faceLms in self.results.multi_face_landmarks:
                #self.mpDraw.draw_landmarks(frame,faceLms,self.mpFaceMesh.FACEMESH_TESSELATION,self.drawSpec,self.drawSpec)#Desenha todos os pontos no rosto
                h,w,c=frame.shape
                points=faceLms.landmark
                face=[]
                for id,point in enumerate(points):#Pega todos os pontos no rosto
                    cx=int(point.x*w)
                    cy=int(point.y*h)
                    #print(f"{id}\n{point}")
                    cv.putText(frame,str(id),(cx,cy),cv.FONT_HERSHEY_PLAIN,0.6,(0,255,0),1)#Escreve o id do landmark no próprio landmark
                    #cv.circle(frame,(cx,cy),3,RED_COLOR,cv.FILLED)#Desenha todos os pontos no rosto
                    face.append([cx,cy])
                faces.append(faces)
        return frame,faces
def main():
    #cap=cv.VideoCapture(r"videos\9.mp4")
    cap=cv.VideoCapture(0,cv.CAP_DSHOW)
    cTime=0
    pTime=0
    detector=FaceMeshDetector()
    while True:
        _,frame=cap.read()
        frame,faces=detector.findFaceMesh(frame)
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