import cv2
import mediapipe as mp
import numpy as np
from pynput.mouse import Controller, Button
import streamlit as st
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Constants
WIDTH, HEIGHT = 640, 480
FRAME_REDUCTION = 100
SMOOTHENING = 7
CLICK_DISTANCE = 40

# Initialize mouse controller
mouse = Controller()

# Initialize audio control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

class HandDetector:
    def __init__(self, maxHands=1, detectionCon=0.7, trackCon=0.7):
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.lmList = []

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img):
        self.lmList = []
        if self.results.multi_hand_landmarks:
            for id, lm in enumerate(self.results.multi_hand_landmarks[0].landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lmList.append([id, cx, cy])
        return self.lmList

    def fingersUp(self):
        if not self.lmList:
            return []
        fingers = []
        if self.lmList[4][1] > self.lmList[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for i in range(1, 5):
            if self.lmList[8 + 4 * (i - 1)][2] < self.lmList[6 + 4 * (i - 1)][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        return fingers

    def findDistance(self, p1, p2, img, draw=True):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 2)
            cv2.circle(img, (x1, y1), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 5, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), cv2.FILLED)
        length = np.hypot(x2 - x1, y2 - y1)
        return length

def main():
    st.title("AI Virtual Mouse using Hand Gestures")
    run = st.checkbox("Run Virtual Mouse")

    if run:
        pTime = 0
        plocX, plocY = 0, 0
        clocX, clocY = 0, 0

        cap = cv2.VideoCapture(0)
        cap.set(3, WIDTH)
        cap.set(4, HEIGHT)
        detector = HandDetector(maxHands=1)
        screenWidth = st.sidebar.slider("Screen Width", 1280, 1920, 1920)
        screenHeight = st.sidebar.slider("Screen Height", 720, 1080, 1080)

        frame_placeholder = st.empty()

        while cap.isOpened():
            success, img = cap.read()
            if not success:
                st.warning("Failed to read from webcam.")
                break

            img = detector.findHands(img)
            lmList = detector.findPosition(img)

            if lmList:
                fingers = detector.fingersUp()
                x1, y1 = lmList[8][1:]  # Index finger tip

                # Move the cursor
                if fingers[1] == 1 and fingers[2] == 0:
                    x3 = np.interp(x1, (FRAME_REDUCTION, WIDTH - FRAME_REDUCTION), (0, screenWidth))
                    y3 = np.interp(y1, (FRAME_REDUCTION, HEIGHT - FRAME_REDUCTION), (0, screenHeight))
                    clocX = plocX + (x3 - plocX) / SMOOTHENING
                    clocY = plocY + (y3 - plocY) / SMOOTHENING
                    mouse.position = (screenWidth - clocX, clocY)
                    plocX, plocY = clocX, clocY

                # Left Click
                if fingers[1] == 1 and fingers[2] == 1:
                    length = detector.findDistance(8, 12, img)
                    if length < CLICK_DISTANCE:
                        mouse.click(Button.left, 1)

                # Right Click: Thumb and Index finger together, all others down
                if fingers[0] == 1 and fingers[1] == 1 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                    length = detector.findDistance(4, 8, img)
                    if length < CLICK_DISTANCE:
                        mouse.click(Button.right, 1)
                        time.sleep(0.3)  # prevent double right-clicks

                # Volume Control
                if fingers[0] == 1 and fingers[1] == 1:
                    length = detector.findDistance(4, 8, img)
                    vol = np.interp(length, [30, 200], [volume.GetVolumeRange()[1], volume.GetVolumeRange()[0]])
                    volume.SetMasterVolumeLevel(vol, None)

            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(img, channels="RGB")

        cap.release()

if __name__ == "__main__":
    main()
