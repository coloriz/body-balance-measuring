from time import time
from threading import Timer

import cv2
import numpy as np
from numpy.linalg import norm
import requests

from helper import Keypoint, Mode, Alignment, put_text
from pose import render_keypoints


HOST = '{{API_HOST}}'

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Failed to initialize video capture!')
    exit(1)

w, h = 1280, 720

ret1 = cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
ret2 = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
# assert ret1 is True and ret2 is True, 'Failed to set frame size'

WINDOW_NAME = 'Tutorial'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

border_margin = round(max(w, h) * 0.025)
state = Mode.Idle
anchor_keypoint = None
measuring_start_time = None
normal_sec = 25
score = 0
start_anchor = None
deviation_threshold = None
score_timeout = None
DEBUG = True
neck_hands = [Keypoint.Neck, Keypoint.RWrist, Keypoint.LWrist]
timer = None


def initialize_params():
    global anchor_keypoint, measuring_start_time, score, start_anchor, deviation_threshold, score_timeout, timer
    anchor_keypoint = None
    measuring_start_time = None
    score = 0
    start_anchor = None
    deviation_threshold = None
    score_timeout = None
    if timer:
        timer.cancel()
    timer = None


def start_measuring(pose_keypoints):
    global measuring_start_time, anchor_keypoint, start_anchor, deviation_threshold, state
    measuring_start_time = time()
    anchor_keypoint = max(Keypoint.LAnkle, Keypoint.RAnkle, key=lambda k: pose_keypoints[k, 1])
    start_anchor = pose_keypoints[anchor_keypoint, :2]
    deviation_threshold = int(norm(pose_keypoints[Keypoint.Nose, :2] - start_anchor) * 0.05)
    state = Mode.Measuring


with requests.Session() as s:
    while True:
        ret, frame = cap.read()
        if not ret:
            print('Failed to read frame!')
            break

        ret, encoded = cv2.imencode('.jpg', frame)
        if not ret:
            break

        try:
            res = s.post(HOST, files={'frame': encoded.tobytes()})
            data = res.json()
        except Exception:
            continue

        if data['code'] == 0:
            keypoints = np.array(data['keypoints'], np.float32)
            pose_keypoints = keypoints[0]
            if DEBUG:
                render_keypoints(frame, keypoints)
        else:
            state = Mode.NotDetected

        put_text(frame, state.name, (border_margin, border_margin), (0, 255, 0))

        if state == Mode.Idle:
            lateral_raised = np.std(pose_keypoints[neck_hands, 1]) < 20
            torso_length = np.abs(pose_keypoints[Keypoint.Neck, 1] - pose_keypoints[Keypoint.MidHip, 1])
            ankle_diff = np.abs(pose_keypoints[Keypoint.LAnkle, 1] - pose_keypoints[Keypoint.RAnkle, 1])
            if lateral_raised and torso_length * 0.2 < ankle_diff:
                if not timer:
                    timer = Timer(0.7, start_measuring, [pose_keypoints])
                    timer.start()
            else:
                if timer:
                    timer.cancel()
                    timer = None
        elif state == Mode.NotDetected:
            initialize_params()
            state = Mode.Idle
        elif state == Mode.Measuring:
            elapsed = time() - measuring_start_time
            score = int(elapsed / normal_sec * 100)
            if elapsed >= normal_sec:
                score_timeout = time()
                state = Mode.Normal
            # Abnormality check
            current_anchor = pose_keypoints[anchor_keypoint, :2]
            deviation = norm(current_anchor - start_anchor)
            if deviation > deviation_threshold:
                score_timeout = time()
                state = Mode.Abnormal
            x, y = current_anchor
            put_text(frame, f'{elapsed:.1f} s', (border_margin, h - border_margin), (255, 255, 255))
            put_text(frame, f'score: {score}', (w - border_margin, h - border_margin), (255, 255, 255),
                     Alignment.RIGHT)
            if DEBUG:
                cv2.circle(frame, tuple(start_anchor), deviation_threshold, (0, 255, 0), 2)
                cv2.circle(frame, (x, y), 10, (0, 0, 255), -1)
                put_text(frame, 'anchor', (int(x) + border_margin, int(y)), (0, 0, 255))
        elif state == Mode.Normal or state == Mode.Abnormal:
            put_text(frame, f'SCORE: {score}', (w // 2, h // 2), (0, 0, 255), Alignment.CENTER, 5, 8, cv2.LINE_AA)
            if time() - score_timeout >= 5:
                initialize_params()
                state = Mode.Idle

        cv2.imshow(WINDOW_NAME, frame)

        pressed_key = cv2.waitKey(1) & 0xFF
        if pressed_key == ord('q'):
            break
        elif pressed_key == ord('r'):
            initialize_params()
            state = Mode.Idle
        elif pressed_key == ord('d'):
            DEBUG = not DEBUG
