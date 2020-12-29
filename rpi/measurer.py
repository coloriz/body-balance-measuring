from enum import Enum, IntEnum, auto
from time import time, sleep
from threading import Thread

import cv2 as cv
import numpy as np
from numpy.linalg import norm
import requests

from helper import Alignment, put_text, fill_alpha_channel


class Mode(Enum):
    Idle = auto()
    NotDetected = auto()
    Measuring = auto()
    Abnormal = auto()
    Normal = auto()


class Keypoint(IntEnum):
    Nose = 0
    Neck = 1
    RShoulder = 2
    RElbow = 3
    RWrist = 4
    LShoulder = 5
    LElbow = 6
    LWrist = 7
    MidHip = 8
    RHip = 9
    RKnee = 10
    RAnkle = 11
    LHip = 12
    LKnee = 13
    LAnkle = 14
    REye = 15
    LEye = 16
    REar = 17
    LEar = 18
    LBigToe = 19
    LSmallToe = 20
    LHeel = 21
    RBigToe = 22
    RSmallToe = 23
    RHeel = 24
    Background = 25


class BodyBalanceMeasurer:
    def __init__(self, processor, config):
        self.config = config
        cam_config = config.picamera
        cam_resolution = cam_config.resolution
        self._terminated = False
        self.processor = processor
        self.resolution = cam_resolution.width, cam_resolution.height
        self.framerate = cam_config.framerate
        self.border_margin = round(max(self.resolution) * 0.025)
        self._text_plate = np.zeros((cam_resolution.height, cam_resolution.width, 4), np.uint8)
        self.text_layer = self._text_plate.copy()

        self.state = Mode.Idle
        self.anchor_keypoint = None
        self.measuring_start_time = None
        self.normal_sec = config.measurer.normal_sec
        self.score = 0
        self.start_anchor = None
        self.deviation_threshold = None
        self.score_popup_timeout = config.measurer.score_popup_timeout
        self.score_timeout = None
        self.neck_hands = [Keypoint.Neck, Keypoint.RWrist, Keypoint.LWrist]
        self.timer = None
        self.elapsed = 0

        self.user_id = ''

        self._worker = Thread(target=self._measure_loop, name='body_balance_measerer')
        self._worker.start()

    def close(self):
        self._terminated = True
        self._worker.join()
    
    def reset(self):
        self.anchor_keypoint = None
        self.measuring_start_time = None
        self.score = 0
        self.elapsed = 0
        self.start_anchor = None
        self.deviation_threshold = None
        self.score_timeout = None
        if self.timer:
            self.timer.cancel()
            self.timer = None
    
    def start_measuring(self, user_id) -> bool:
        """시작 성공 여부를 True/False로 반환. 측정 시작했는데 keypoints가 없는 경우 시작 불가"""
        # TODO: 측정 시작했는데 pose_keypoints가 None인 경우 처리
        pose_keypoints = self.processor.keypoints
        if pose_keypoints is None:
            return False
        pose_keypoints = pose_keypoints[0]
        self.user_id = user_id
        self.measuring_start_time = time()
        self.anchor_keypoint = max(Keypoint.LAnkle, Keypoint.RAnkle, key=lambda k: pose_keypoints[k, 1])
        self.start_anchor = pose_keypoints[self.anchor_keypoint, :2]
        self.deviation_threshold = int(norm(pose_keypoints[Keypoint.Nose, :2] - self.start_anchor) * 0.05)
        self.state = Mode.Measuring
        return True

    def _measure_loop(self):
        config = self.config
        w, h = self.resolution
        frame_interval = 1 / self.framerate
        plate = self._text_plate
        border_margin = self.border_margin

        while not self._terminated:
            start = time()
            pose_keypoints = self.processor.keypoints

            if pose_keypoints is None:
                self.state = Mode.NotDetected
            else:
                pose_keypoints = pose_keypoints[0]
            
            plate.fill(0)
            put_text(plate, self.state.name, (border_margin, border_margin), (0, 255, 0))

            if self.state == Mode.Idle:
                pass
            elif self.state == Mode.NotDetected:
                self.state = Mode.Idle
            elif self.state == Mode.Measuring:
                self.elapsed = time() - self.measuring_start_time
                self.score = int(self.elapsed / self.normal_sec * 100)
                if self.elapsed >= self.normal_sec:
                    self.score_timeout = time()
                    self.state = Mode.Normal
                # check abnormality
                current_anchor = pose_keypoints[self.anchor_keypoint, :2]
                deviation = norm(current_anchor - self.start_anchor)
                if deviation > self.deviation_threshold:
                    self.score_timeout = time()
                    self.state = Mode.Abnormal
                x, y = current_anchor
                put_text(plate, f'{self.elapsed:.1f} s / id: {self.user_id}', (border_margin, h - border_margin), (255, 255, 255))
                put_text(plate, f'score: {self.score}', (w - border_margin, h - border_margin), (255, 255, 255), Alignment.RIGHT)

                cv.circle(plate, tuple(self.start_anchor), self.deviation_threshold, (0, 255, 0), 2)
                cv.circle(plate, (x, y), 10, (0, 0, 255), -1)
                put_text(plate, 'anchor', (int(x) + border_margin, int(y)), (0, 0, 255))
            elif self.state == Mode.Normal or self.state == Mode.Abnormal:
                put_text(plate, f'SCORE: {self.score}', (w // 2, h // 2), (0, 0, 255), Alignment.CENTER, 5, 8, cv.LINE_AA)
                if time() - self.score_timeout >= self.score_popup_timeout:
                    params = {
                        'user_seq': self.user_id,
                        'module_type': config.database.module_type,
                        'score_1': self.score,
                        'score_2': f'{self.elapsed:.3f}'
                    }
                    try:
                        res = requests.post(f'{config.database.url}/scores/save', json=params, timeout=5)
                        res.raise_for_status()
                    except requests.exceptions.Timeout:
                        print('Timeout error raised while posting data to database')
                    except requests.exceptions.HTTPError as e:
                        print(f'Unsuccessful status code: {e}')
                    self.reset()
                    self.state = Mode.Idle
            
            fill_alpha_channel(plate)
            self.text_layer = plate.copy()

            elapsed = time() - start
            if elapsed < frame_interval:
                sleep(frame_interval - elapsed)
