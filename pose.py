import math
from typing import Tuple

import cv2
import numpy as np

thickness_circle_ratio = 1 / 75
thickness_line_ratio_wrt_circle = 0.75
BODY_25_PAIRS = [(1, 8), (1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10),
                 (10, 11), (8, 12), (12, 13), (13, 14), (1, 0), (0, 15), (15, 17),
                 (0, 16), (16, 18), (14, 19), (19, 20), (14, 21), (11, 22), (22, 23), (11, 24)]
BODY_25_SCALES = [1]
BODY_25_COLORS = [(255, 0, 85), (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0),
                  (0, 255, 0), (255, 0, 0), (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
                  (0, 0, 255), (255, 0, 170), (170, 0, 255), (255, 0, 255), (85, 0, 255), (0, 0, 255), (0, 0, 255),
                  (0, 0, 255), (0, 255, 255), (0, 255, 255), (0, 255, 255)]


def get_keypoints_rectangle(person: np.ndarray, threshold) -> Tuple:
    type_info = np.finfo(person.dtype)
    min_x = type_info.max
    max_x = type_info.min
    min_y = min_x
    max_y = max_x

    for part in person:
        score = part[2]
        if score > threshold:
            x, y = part[:2]
            # Set X
            if max_x < x: max_x = x
            if min_x > x: min_x = x
            # Set Y
            if max_y < y: max_y = y
            if min_y > y: min_y = y

    return min_x, min_y, max_x - min_x, max_y - min_y if max_x >= min_x and max_y >= min_y else (0, 0, 0, 0)


def render_keypoints(frame: np.ndarray, keypoints: np.ndarray, threshold=0.05):
    height, width = frame.shape[:2]
    area = width * height
    # frame_bgr = frame.astype(np.float32)

    # Parameters
    number_scales = len(BODY_25_SCALES)
    threshold_rectangle = 0.1

    # Keypoints
    for person in keypoints:
        tmp = get_keypoints_rectangle(person, threshold_rectangle)
        x, y, w, h = get_keypoints_rectangle(person, threshold_rectangle)
        if w * h > 0:
            ratio_area = min(1, max(w / width, h / height))
            # Size-dependant variables
            thickness_ratio = max(round(math.sqrt(area) * thickness_circle_ratio * ratio_area), 2)
            thickness_circle = max(1, thickness_ratio if ratio_area > 0.05 else -1)
            thickness_line = max(1, round(thickness_ratio * thickness_line_ratio_wrt_circle))
            radius = thickness_ratio / 2

            # Draw lines
            for i, j in BODY_25_PAIRS:
                kp1, kp2 = person[i], person[j]
                if kp1[2] > threshold and kp2[2] > threshold:
                    thickness_line_scaled = int(round(thickness_line * BODY_25_SCALES[j % number_scales]))
                    color = BODY_25_COLORS[j][::-1]
                    p1 = int(round(kp1[0])), int(round(kp1[1]))
                    p2 = int(round(kp2[0])), int(round(kp2[1]))
                    cv2.line(frame, p1, p2, color, thickness_line_scaled)

            # Draw Circles
            for i, part in enumerate(person):
                if part[2] > threshold:
                    radius_scaled = int(round(radius * BODY_25_SCALES[i % number_scales]))
                    thickness_circle_scaled = int(round(thickness_circle * BODY_25_SCALES[i % number_scales]))
                    color = BODY_25_COLORS[i][::-1]
                    center = int(round(part[0])), int(round(part[1]))
                    cv2.circle(frame, center, radius_scaled, color, thickness_circle_scaled)


def render_keypoints2(frame: np.ndarray, keypoints: np.ndarray, threshold=0.05):
    height, width = frame.shape[:2]
    area = width * height
    # frame_bgr = frame.astype(np.float32)

    # Parameters
    number_scales = len(BODY_25_SCALES)
    threshold_rectangle = 0.1

    # Keypoints
    for person in keypoints:
        tmp = get_keypoints_rectangle(person, threshold_rectangle)
        x, y, w, h = get_keypoints_rectangle(person, threshold_rectangle)
        if w * h > 0:
            ratio_area = min(1, max(w / width, h / height))
            # Size-dependant variables
            thickness_ratio = max(round(math.sqrt(area) * thickness_circle_ratio * ratio_area), 2)
            thickness_circle = max(1, thickness_ratio if ratio_area > 0.05 else -1)
            thickness_line = max(1, round(thickness_ratio * thickness_line_ratio_wrt_circle))
            radius = thickness_ratio / 2

            # Draw lines
            for i, j in BODY_25_PAIRS:
                kp1, kp2 = person[i], person[j]
                if kp1[2] > threshold and kp2[2] > threshold:
                    thickness_line_scaled = int(round(thickness_line * BODY_25_SCALES[j % number_scales]))
                    color = BODY_25_COLORS[j][::-1]
                    color = color[0], color[1], color[2], 255
                    p1 = int(round(kp1[0])), int(round(kp1[1]))
                    p2 = int(round(kp2[0])), int(round(kp2[1]))
                    cv2.line(frame, p1, p2, color, thickness_line_scaled)

            # Draw Circles
            for i, part in enumerate(person):
                if part[2] > threshold:
                    radius_scaled = int(round(radius * BODY_25_SCALES[i % number_scales]))
                    thickness_circle_scaled = int(round(thickness_circle * BODY_25_SCALES[i % number_scales]))
                    color = BODY_25_COLORS[i][::-1]
                    color = color[0], color[1], color[2], 255
                    center = int(round(part[0])), int(round(part[1]))
                    cv2.circle(frame, center, radius_scaled, color, thickness_circle_scaled)
