import cv2
import mediapipe as mp
import time
import numpy as np
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

wCam, hCam = 640, 480

def angle(p1, p2, p3):
    ang = math.degrees(math.atan2(p3[1]-p2[1], p3[0]-p2[0])-math.atan2(p1[1]-p2[1], p1[0]-p2[0]))
    return ang
class poseDetector():
    def __init__(self,
                    static_image_mode = False,
                    model_complexity = 1,
                    smooth_landmarks = True,
                    enable_segmentation = False,
                    smooth_segmentation = True,
                    min_detection_confidence = 0.5,
                    min_tracking_confidence = 0.5):

        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.static_image_mode,
                                     self.model_complexity,
                                     self.smooth_landmarks,
                                     self.enable_segmentation,
                                     self.smooth_segmentation,
                                     self.min_detection_confidence,
                                     self.min_tracking_confidence)

    def detect(self, frame, draw=False, radius=0, color=(0,0,0), thickness=0):
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(RGBframe)
        self.points = []
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                if (11 <= id <= 16) or id == 23 or id == 24:
                    if draw:
                        cv2.circle(frame, (cx, cy), radius, color, thickness)
                    self.points.append((cx, cy))
            if draw:
                for i, cor in enumerate(self.points):
                    if 1 < i < 6:
                        cv2.line(frame, (cor[0], cor[1]), (self.points[i-2][0], self.points[i-2][1]), (0, 0, 0), 4)
                    else:
                        if i == 0:
                            cv2.line(frame, (cor[0], cor[1]), (self.points[1][0], self.points[1][1]), (0, 0, 0), 4)
                            cv2.line(frame, (cor[0], cor[1]), (self.points[6][0], self.points[6][1]), (0, 0, 0), 4)
                        if i == 1:
                            cv2.line(frame, (cor[0], cor[1]), (self.points[7][0], self.points[7][1]), (0, 0, 0), 4)
                        if i == 6:
                            cv2.line(frame, (cor[0], cor[1]), (self.points[7][0], self.points[7][1]), (0, 0, 0), 4)
                for i, cor in enumerate(self.points):
                    cv2.circle(frame, (cor[0], cor[1]), radius, color, thickness)
                    cv2.circle(frame, (cor[0], cor[1]), radius+5, color, 2)

        return frame

    def numPose(self):
        if self.results.pose_landmarks != None:
            return len(self.results.pose_landmarks)
        else:
            return 0

    def getPose(self):
        return self.points

class faceDetector():
    def __init__(self, minDetectionConfidence=0.75):
        self.mpFace = mp.solutions.face_detection
        self.faces = self.mpFace.FaceDetection(minDetectionConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def detect(self, frame, draw=False, color=(0,0,0), thickness=0):
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.faces.process(RGBframe)
        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                per = str(int(detection.score[0] * 100)) + '%'
                h, w, c = frame.shape
                f = detection.location_data.relative_bounding_box
                x, y, cw, ch = int(f.xmin * w), int(f.ymin * h), int(f.width * w), int(f.height * h)
                if draw:
                    x -= 10
                    cw += 10
                    y -= 20
                    ch += 25
                    cv2.rectangle(frame, (x - 3, y - 3), (x + cw + 3, y + ch + 3), color, thickness)
                    cv2.putText(frame, per, (x - 2, y - 10), \
                                cv2.FONT_HERSHEY_SIMPLEX, int(int(cw*0.2)/20), color, thickness+1)

                    cv2.line(frame, (x - 3, y - 3), (x - 3 + int(cw * 0.2), y - 3), (255, 0, 0), thickness + 3)
                    cv2.line(frame, (x - 3, y - 3), (x - 3, y - 3 + int(ch * 0.2)), (255, 0, 0), thickness + 3)

                    cv2.line(frame,(x - 3, y + ch + 3),(x - 3 + int(cw * 0.2), y + ch + 3), (255, 0, 0), thickness + 3)
                    cv2.line(frame,(x - 3, y + ch + 3),(x - 3, y + ch + 3 - int(ch * 0.2)), (255, 0, 0), thickness + 3)

                    cv2.line(frame,(x + cw + 3, y - 3),(x + cw + 3 - int(cw * 0.2), y - 3), (255, 0, 0), thickness + 3)
                    cv2.line(frame,(x + cw + 3, y - 3),(x + cw + 3, y - 3 + int(ch * 0.2)), (255, 0, 0), thickness + 3)

                    cv2.line(frame, (x + cw + 3, y + ch + 3),\
                             (x + cw + 3 - int(cw * 0.2), y + ch + 3), (255, 0, 0), thickness + 3)
                    cv2.line(frame, (x + cw + 3, y + ch + 3),\
                             (x + cw + 3, y + ch + 3 - int(ch * 0.2)), (255, 0, 0), thickness + 3)
        return frame

    def getFaces(self):
        res=[]
        faces = self.results.detections
        if faces:
            for id, detection in enumerate(faces):
                res.append(detection)
        return res


class faceMeshing():
    def __init__(self, mode=False, maxFaces=3, refineLandmarks=False, detectionCon=0.5, trackingCon=0.5):
        self.mpMesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.meshs = self.mpMesh.FaceMesh(mode, maxFaces, refineLandmarks, detectionCon, trackingCon)

    def draw(self, frame, color, thickness=1, radius=0):
        self.spec = self.mpDraw.DrawingSpec(color, thickness, radius)
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.meshs.process(RGBframe)
        if results.multi_face_landmarks:
            for id, face in enumerate(results.multi_face_landmarks):
                for cor in face.landmark:
                    h, w, c = frame.shape
                    cx, cy = int(cor.x * w), int(cor.y * h)
                self.mpDraw.draw_landmarks(frame, face, self.mpMesh.FACEMESH_TESSELATION, self.spec, self.spec)
        return frame


class handDetector():
    def __init__ (self, mode=False, mxHands=2, modelCom=1, detectConf=0.5, trackConf=0.5):
        self.mode =mode
        self.mxHands=mxHands
        self.detectConf=detectConf
        self.trackConf=trackConf
        self.modelCom = modelCom
        self.Draw = mp.solutions.drawing_utils
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.mxHands, self.modelCom, self.detectConf, self.trackConf)


    def detect(self, frame, draw=False, radius=0, thickness=0, color=(0,0,0)):
        RGBframe = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(RGBframe)
        self.num = 0
        if self.results.multi_hand_landmarks != None:
            self.num = len(self.results.multi_hand_landmarks)
        if self.results.multi_hand_landmarks:
            for hand in self.results.multi_hand_landmarks:
                if draw:
                    self.Draw.draw_landmarks(frame, hand, self.mpHands.HAND_CONNECTIONS)
                for id, cor in enumerate(hand.landmark):
                    h, w, c = frame.shape
                    x, y = int(cor.x * w), int(cor.y * h)
                    if draw:
                        cv2.circle(frame, (x, y), radius, color, thickness)
        return frame

    def numHands(self):
        return self.num

    def getHands(self, frame, handID):
        res=[]
        if self.num>=handID+1:
            if self.results.multi_hand_landmarks != None:
                hand = self.results.multi_hand_landmarks[handID]
                if hand:
                    for id, cor in enumerate(hand.landmark):
                        h, w, c = frame.shape
                        #print(cor)
                        x, y = int(cor.x * w), int(cor.y * h)
                        z= cor.z
                        res.append((id,x,y,z))
        return res
    def fingersUp(self, nodes):
        fingers = [0, 0, 0, 0, 0]
        if nodes:
            theta = math.pi / 2
            try:
                theta = math.atan((nodes[13][1] - nodes[0][1]) / (nodes[13][2] - nodes[0][2]))
            except ZeroDivisionError:
                pass
            if nodes[13][2] > nodes[0][2]:
                theta -= math.pi

            s = math.sin(theta)
            c = math.cos(theta)
            for nodeId, node in enumerate(nodes):
                nodeX = node[1]
                nodeY = node[2]
                # print(nodes[0])
                nodeX -= nodes[0][1]
                nodeY -= nodes[0][2]
                newNodeX = int(nodeX * c - nodeY * s)
                newNodeY = int(nodeX * s + nodeY * c)
                newNodeX += nodes[0][1]
                newNodeY += nodes[0][2]
                nodes[nodeId] = (nodeId, newNodeX, newNodeY, 0)
                # frame = cv.circle(frame, (newNodeX, newNodeY), 5, (255, 255, 255), 10)

            if nodes[5][1] > nodes[17][1]:
                for nodeId, node in enumerate(nodes):
                    nodeX = node[1]
                    nodeY = node[2]
                    nodes[nodeId] = (nodeId, 2 * nodes[0][1] - nodeX, nodeY, 0)

            thresh = -40
            if nodes[4][1] < nodes[2][1] + thresh:
                fingers[0] = 1
            if nodes[8][2] < nodes[5][2] + thresh:
                fingers[1] = 2
            if nodes[12][2] < nodes[9][2] + thresh:
                fingers[2] = 4
            if nodes[16][2] < nodes[13][2] + thresh:
                fingers[3] = 8
            if nodes[20][2] < nodes[17][2] + thresh:
                fingers[4] = 16
        return fingers

###########################################
#               HAND DETECTOR             #
###########################################

def mainhd(cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)):
    wCam, hCam = 640, 480
    cap.set(3, wCam)
    cap.set(4, hCam)
    poseD = poseDetector()
    Detector = handDetector(mxHands=1, detectConf=0.7)
    while cap.isOpened():
        case = 0
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        if ret:
            frame = Detector.detect(frame, True, 7, -1, (255, 0, 0))
            frame = poseD.detect(frame)
            poselm = poseD.getPose()
            if len(poselm) > 5:
                if poselm[4][1] < poselm[0][1] - 50 and poselm[2][1] < poselm[0][1] - 50 \
                        and poselm[3][1] < poselm[0][1] - 50 and poselm[5][1] < poselm[0][1] - 50:
                    return
            cv2.imshow('camera', frame)
            fingers = []
            for i in range(Detector.numHands()):
                lmlist = Detector.getHands(frame, i)
                fingers = Detector.fingersUp(lmlist)
                for i in Detector.fingersUp(lmlist):
                    case += i
            case = int(case)
            if case == 17:
                mainvc(cap)
            if case == 28:
                maindr(cap)
        key = cv2.waitKey(1)
        if key == 27 or key == ord('q'):
            break
    cv2.destroyAllWindows()
    cap.release()

###########################################
#             FACE AND MESH               #
###########################################
def mainfm(cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)):
    wCam, hCam = 640, 480
    cap.set(3, wCam)
    cap.set(4, hCam)
    poseD = poseDetector()
    mesh = faceMeshing()
    Detector = faceDetector()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame = Detector.detect(frame, True, (255, 0, 0), 1)
            frame = mesh.draw(frame, (0, 255, 0))
            frame = poseD.detect(frame)
            poselm = poseD.getPose()
            if len(poselm) > 5:
                if poselm[4][1] < poselm[0][1] - 50 and poselm[2][1] < poselm[0][1] - 50 \
                        and poselm[3][1] < poselm[0][1] - 50 and poselm[5][1] < poselm[0][1] - 50:
                    return
            cv2.imshow('camera', frame)
            if cv2.waitKey(1) == 27:
                break
    cv2.destroyAllWindows()
    cap.release()

###########################################
#              VOLUME CONTROL             #
###########################################
def mainvc(cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)):
    cap.set(3, wCam)
    cap.set(4, hCam)
    pTime = 0
    detector = handDetector(detectConf=0.7)
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(
        IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume = cast(interface, POINTER(IAudioEndpointVolume))
    # volume.GetMute()
    # volume.GetMasterVolumeLevel()
    volRange = volume.GetVolumeRange()
    minVol = volRange[0]
    maxVol = volRange[1]
    #print(minVol, maxVol)
    vol = 0
    volBar = 400
    volPer = 0
    count = 300
    check = False
    color = (0, 0, 0)
    while True:
        cTime = time.time()
        fps = int(1 / (cTime - pTime))
        pTime = cTime
        success, img = cap.read()
        img = cv2.flip(img, 1)
        img = detector.detect(img, True, 2, -1, (255, 0, 255))
        count += 1
        num = detector.numHands()
        lmList = detector.getHands(img, 1)
        lmList2 = None
        if check:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)
        lenght1 = 50
        lmList2 = detector.getHands(img, 0)
        if lmList2:
            xx1, yy1 = lmList2[8][1], lmList2[8][2]
            fingers = detector.fingersUp(lmList2)
            case = 0
            for i in fingers:
                case += i
            case = int(case)
            if case == 6:
                return
            cv2.circle(img, (xx1, yy1), 10, (255, 0, 255), cv2.FILLED)
            if xx1 >= 30 and xx1 <= 90 and yy1 >= 20 and yy1 <= 80 and count >= fps * 3:
                if check:
                    count = 0
                    check = False
                    color = (0, 0, 255)
                else:
                    count = 0
                    check = True
                    color = (0, 255, 0)
        if check:
            if lmList:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                x3, y3 = lmList[0][1], lmList[0][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                ang = abs(angle((x1, y1), (x3, y3), (x2, y2)))
                if ang >= 100:
                    ang = 360 - ang
                # print(ang)
                cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)

                vol = np.interp(ang, [5, 65], [minVol, maxVol])
                volBar = np.interp(vol, [minVol, maxVol], [400, 150])
                volPer = np.interp(vol, [minVol, maxVol], [0, 100])
                volume.SetMasterVolumeLevel(vol, None)
                # print(vol)
                if ang <= 5:
                    cv2.circle(img, (cx, cy), 15, (0, 255, 0), cv2.FILLED)

                cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0), 3)
                cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
                cv2.putText(img, f'{int(volPer)} %', (40, 450), cv2.FONT_HERSHEY_COMPLEX,
                            1, (255, 0, 0), 3)
        text = 'OFF'
        if check:
            text = ' ON'

        cv2.circle(img, (60, 50), 40, color, -1)
        cv2.putText(img, text, (28, 60), cv2.FONT_HERSHEY_COMPLEX,
                    1, (0, 0, 0), 3)
        cv2.imshow("camera", img)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()

###########################################
#                 DRAWING                 #
###########################################
def maindr(cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)):
    wCam, hCam = 640, 480
    cap.set(3, wCam)
    cap.set(4, hCam)
    detector = handDetector(mxHands=1, detectConf=0.75)
    img = np.zeros((hCam, wCam, 3), np.uint8)
    Cimg = np.zeros((hCam, wCam, 3), np.uint8)
    _, Cimg = cv2.threshold(Cimg, 200, 255, cv2.THRESH_BINARY_INV)
    cx, cy = None, None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            fingers = []
            frame = cv2.flip(frame, 1)
            frame = detector.detect(frame)
            hand = detector.getHands(frame, 0)
            if hand:
                fingers = detector.fingersUp(hand)
            hand = detector.getHands(frame, 0)
            if hand:
                case = 0
                for i in fingers:
                    case += i
                case = int(case)
                if case == 6:
                    return
                if case == 1:
                    cv2.circle(img, (hand[4][1], hand[4][2]), 20, (0, 0, 0), -1)
                    cv2.circle(Cimg, (hand[4][1], hand[4][2]), 20, (255, 255, 255), -1)
                    cv2.circle(frame, (hand[4][1], hand[4][2]), 20, (252, 252, 3), -1)
                    if cx is not None and cy is not None:
                        cv2.line(img, (hand[4][1], hand[4][2]), (cx, cy), (0, 0, 0), 30)
                        cv2.line(Cimg, (hand[4][1], hand[4][2]), (cx, cy), (255, 255, 255), 30)
                        cx = hand[4][1]
                        cy = hand[4][2]
                    else:
                        cx = hand[4][1]
                        cy = hand[4][2]
                if case == 2:
                    cv2.circle(img, (hand[8][1], hand[8][2]), 10, (255, 255, 255), -1)
                    cv2.circle(Cimg, (hand[8][1], hand[8][2]), 10, (255, 0, 0), -1)
                    if cx is not None and cy is not None:
                        cv2.line(img, (hand[8][1], hand[8][2]), (cx, cy), (255, 255, 255), 17)
                        cv2.line(Cimg, (hand[8][1], hand[8][2]), (cx, cy), (255, 0, 0), 17)
                    cx = hand[8][1]
                    cy = hand[8][2]
                if case == 0:
                    cx = None
                    cy = None
            frame = cv2.bitwise_or(frame, img)
            frame = cv2.bitwise_and(frame, Cimg)
            cv2.imshow('camera', frame)
        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()
    cap.release()

###########################################
#                  MAIN                   #
###########################################
def main():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    wCam, hCam = 640, 480
    cap.set(3, wCam)
    cap.set(4, hCam)
    poseD = poseDetector()
    count = 0
    timer = 0
    check = False
    pTime = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            ORGframe = frame.copy()
            ORGframe = cv2.flip(ORGframe, 1)
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime
            if check:
                timer += 1
            frame = cv2.flip(frame, 1)
            frame = poseD.detect(frame, True, 13, (255, 0, 0), -1)
            poselm = poseD.getPose()
            if len(poselm) > 5:
                if poselm[4][1] < poselm[0][1] - 50 and poselm[2][1] < poselm[0][1] - 50 \
                        and poselm[3][1] > poselm[0][1] and poselm[5][1] > poselm[0][1] :
                    mainfm(cap)
                if poselm[4][1] > poselm[0][1] and poselm[2][1] > poselm[0][1] and poselm[3][1] < poselm[0][1] - 50 \
                        and poselm[5][1] < poselm[0][1] - 50:
                    mainhd(cap)
            if len(poselm) > 7:
                if poselm[1][0] < poselm[4][0] < poselm[0][0] and poselm[1][0] < poselm[5][0] < poselm[0][0] \
                        and poselm[1][1] < poselm[4][1] < poselm[6][1] and poselm[1][1] < poselm[5][1] < poselm[6][1] \
                        and poselm[4][0] < poselm[5][0]:
                    check = True
                    timer = 0
            if 0 < timer < fps:
                cv2.putText(frame, '1', (int(wCam/2) - 80, int(hCam/2) + 50), \
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 7, (175, 11, 224), 20)
            if fps*1 < timer < fps*2:
                cv2.putText(frame, '2', (int(wCam/2) - 80, int(hCam/2) + 50), \
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 7, (175, 11, 224), 20)
            if fps*2 < timer < fps*3:
                cv2.putText(frame, '3', (int(wCam/2) - 80, int(hCam/2) + 50), \
                            cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 7, (175, 11, 224), 20)
            if check and timer >= int(fps*3):
                count += 1
                screen = 'photo_' + str(count) + '.jpg'
                cv2.imwrite(screen, ORGframe)
                check = False
                timer = 0
            cv2.imshow('camera', frame)
            if cv2.waitKey(1) == 27:
                break
    cv2.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
   main()