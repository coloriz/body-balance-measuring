from threading import Thread
from time import time, sleep

from picamera import PiCamera
import numpy as np

from processors import FrameProcessor
import pose
from measurer import BodyBalanceMeasurer
from helper import fill_alpha_channel


class MainController:
    def __init__(self, config):
        cam_config = config.picamera
        cam_resolution = cam_config.resolution
        self.cam = PiCamera()
        self.cam.resolution = cam_resolution.width, cam_resolution.height
        self.cam.framerate = cam_config.framerate
        self.cam.rotation = cam_config.rotation
        self.cam.start_preview()
        self.output = FrameProcessor(config.server_url, config.workers)
        self.cam.start_recording(self.output, format='mjpeg')
        self._rendered_keypoints_timestamp = time()
        self._keypoints_drawing = np.zeros((cam_resolution.height, cam_resolution.width, 4), np.uint8)
        self._keypoints_overlay = self.cam.add_overlay(self._keypoints_drawing, layer=3, format='bgra')
        
        self._terminated = False
        self.bbm = BodyBalanceMeasurer(self.output, config)
        self._text_overlay = self.cam.add_overlay(self.bbm.text_layer, layer=4, format='bgra')
        self._overlays_update_thread = Thread(target=self._update_overlays, name='overlay_updater')
        self._overlays_update_thread.start()

    def _update_overlays(self):
        # 문서에서 카메라의 framerate보다 더 빠르게 업데이트하지 말라고 언급.
        # 오버레이 갱신 주기 제한
        frame_interval = 1 / int(self.cam.framerate)

        while not self._terminated:
            start = time()
            self._update_keypoints_overlay()
            self._update_text_overlay()
            elapsed = time() - start
            if elapsed < frame_interval:
                sleep(frame_interval - elapsed)
    
    def _update_keypoints_overlay(self):
        # 더 최신의 keypoints가 있는 경우에만 업데이트
        keypoints_timestamp, keypoints = self.output.timestamp_and_keypoints
        if self._rendered_keypoints_timestamp >= keypoints_timestamp:
            return
        self._keypoints_drawing.fill(0)
        if keypoints is not None:
            pose.render_keypoints(self._keypoints_drawing, keypoints)
            fill_alpha_channel(self._keypoints_drawing)
        self._keypoints_overlay.update(self._keypoints_drawing)
        self._rendered_keypoints_timestamp = keypoints_timestamp

    def _update_text_overlay(self):
        self._text_overlay.update(self.bbm.text_layer)
    
    def close(self):
        self._terminated = True
        self._overlays_update_thread.join()
        self.cam.remove_overlay(self._text_overlay)
        self.cam.remove_overlay(self._keypoints_overlay)
        self.cam.stop_recording()
        self.cam.stop_preview()
        self.cam.close()
            