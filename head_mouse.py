import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

MODEL_PATH = "face_landmarker.task"

BASE_SMOOTH = 5
FAST_SMOOTH = 2
DEAD_ZONE = 4
SENSITIVITY = 1
frame_skip = 2

SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.FAILSAFE = False
pyautogui.PAUSE = 0


BaseOptions = mp.tasks.BaseOptions
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


prev_x, prev_y = SCREEN_W/2, SCREEN_H/2
blink_start = 0
dragging = False

def smooth(prev, target, smooth):
    return prev + (target - prev)/smooth

with FaceLandmarker.create_from_options(options) as landmarker:
    frame_skip = 2
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        frame = cv2.flip(frame,1)
        h,w,_ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_img = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        ts = int(time.time()*1000)
        res = landmarker.detect_for_video(mp_img, ts)

        if res.face_landmarks:

            lm = res.face_landmarks[0]
            nose = lm[1]
            cam_x = nose.x * w
            cam_y = nose.y * h
            nx = nose.x
            ny = nose.y

            TRACK_MIN_X = 0.35
            TRACK_MAX_X = 0.65
            TRACK_MIN_Y = 0.35
            TRACK_MAX_Y = 0.60

            nx = np.clip(nx, TRACK_MIN_X, TRACK_MAX_X)
            ny = np.clip(ny, TRACK_MIN_Y, TRACK_MAX_Y)

            screen_x = np.interp(nx, [TRACK_MIN_X, TRACK_MAX_X], [0, SCREEN_W])
            screen_y = np.interp(ny, [TRACK_MIN_Y, TRACK_MAX_Y], [0, SCREEN_H])

            screen_x = prev_x + (screen_x - prev_x)*SENSITIVITY
            screen_y = prev_y + (screen_y - prev_y)*SENSITIVITY

            if abs(screen_x - prev_x) < DEAD_ZONE:
                screen_x = prev_x
            if abs(screen_y - prev_y) < DEAD_ZONE:
                screen_y = prev_y

            speed = abs(screen_x-prev_x) + abs(screen_y-prev_y)
            smooth_factor = FAST_SMOOTH if speed > 40 else BASE_SMOOTH

            curr_x = smooth(prev_x, screen_x, smooth_factor)
            curr_y = smooth(prev_y, screen_y, smooth_factor)

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            eye = abs(lm[159].y - lm[145].y)

            if eye < 0.012:
                if blink_start == 0:
                    blink_start = time.time()
            else:
                if blink_start != 0:
                    duration = time.time() - blink_start
                    if duration > 0.6:
                        pyautogui.rightClick()
                    else:
                        pyautogui.click()
                blink_start = 0

            mouth = abs(lm[13].y - lm[14].y)

            if mouth > 0.035:
                if not dragging:
                    pyautogui.mouseDown()
                    dragging = True
            else:
                if dragging:
                    pyautogui.mouseUp()
                    dragging = False

            cv2.circle(frame,(int(cam_x),int(cam_y)),5,(0,255,0),-1)

        cv2.imshow("PRO V2 Head Mouse", frame)

        if cv2.waitKey(1)==27:
            break

cap.release()
cv2.destroyAllWindows()
