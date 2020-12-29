from enum import IntEnum, auto

import cv2 as cv
import numpy as np
from picamera import PiCamera


class Alignment(IntEnum):
    LEFT = auto()
    CENTER = auto()
    RIGHT = auto()


class OverlayUpdater:
    def __init__(self, camera: PiCamera, front_layer: int, back_layer: int, **kwargs):
        self._camera = camera
        self._front_layer = front_layer
        self._back_layer = back_layer
        self._flag = True
        self._kwargs = kwargs
        self._overlay = None
    
    def update(self, source):
        past_overlay = self._overlay
        layer_num = self._front_layer if self._flag else self._back_layer
        self._overlay = self._camera.add_overlay(source, layer=layer_num, **self._kwargs)
        self._flag = not self._flag
        if past_overlay:
            self._camera.remove_overlay(past_overlay)
    
    def close(self):
        if self._overlay:
            self._camera.remove_overlay(self._overlay)


def put_text(img, text, position, color, alignment=Alignment.LEFT, scale=None, thickness=None, line_type=cv.LINE_8):
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = scale or 0.8
    font_thickness = thickness or 2
    shadow_offset = 2
    text_size, baseline = cv.getTextSize(text, font, font_scale, font_thickness)
    if alignment == Alignment.LEFT:
        final_position = (position[0], position[1] + text_size[1] // 2)
    elif alignment == Alignment.CENTER:
        final_position = (position[0] - text_size[0] // 2, position[1] + text_size[1] // 2)
    else:
        final_position = (position[0] - text_size[0], position[1] + text_size[1] // 2)

    cv.putText(img, text, (final_position[0] + shadow_offset, final_position[1] + shadow_offset),
                font, font_scale, (0, 0, 0), font_thickness, line_type)
    cv.putText(img, text, final_position, font, font_scale, color, font_thickness, line_type)


def fill_alpha_channel(img: np.ndarray) -> np.ndarray:
    mask = np.logical_or(np.logical_or(img[:, :, 0], img[:, :, 1]), img[:, :, 2])
    alpha = img[:, :, 3]
    alpha[mask] = 255
    return img
