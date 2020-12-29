from enum import IntEnum, auto

import cv2


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


class Mode(IntEnum):
    Idle = auto()
    NotDetected = auto()
    Measuring = auto()
    Abnormal = auto()
    Normal = auto()


class Alignment(IntEnum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


def put_text(img, text, position, color, alignment=Alignment.LEFT, scale=None, thickness=None, line_type=cv2.LINE_8):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = scale or 0.8
    font_thickness = thickness or 2
    shadow_offset = 2
    text_size, baseline = cv2.getTextSize(text, font, font_scale, font_thickness)
    if alignment == Alignment.LEFT:
        final_position = (position[0], position[1] + text_size[1] // 2)
    elif alignment == Alignment.CENTER:
        final_position = (position[0] - text_size[0] // 2, position[1] + text_size[1] // 2)
    else:
        final_position = (position[0] - text_size[0], position[1] + text_size[1] // 2)

    cv2.putText(img, text, (final_position[0] + shadow_offset, final_position[1] + shadow_offset),
                font, font_scale, (0, 0, 0, 255), font_thickness, line_type)
    cv2.putText(img, text, final_position, font, font_scale, color, font_thickness, line_type)
