from collections import deque
from copy import deepcopy
from threading import Thread, Lock, Condition
from time import time, sleep
from typing import Optional, Tuple
from io import BytesIO

import requests
from simplejson.errors import JSONDecodeError
import numpy as np


class KeypointsExtractor(Thread):
    def __init__(self, server_url, owner, name, reqeust_timeout=1):
        super(KeypointsExtractor, self).__init__()
        self.name = name
        self.terminated = False
        self.url = server_url
        self.owner = owner
        self.sess = requests.Session()
        self.reqeust_timeout = reqeust_timeout
        self.start()
    
    def run(self):
        sess = self.sess
        owner = self.owner
        url = self.url
        timeout = self.reqeust_timeout

        while not self.terminated:
            with owner._condition:
                owner._condition.wait_for(lambda: owner._recent_frame[1] is not None)
                timestamp, frame = owner._recent_frame
                owner._recent_frame = timestamp, None

            try:
                res = sess.post(url, files={'frame': frame}, timeout=timeout)
                data = res.json()
            except (requests.exceptions.Timeout, JSONDecodeError) as e:
                print(self.name, e)
                continue
    
            keypoints = np.array(data['keypoints'], np.float32) if data['code'] == 0 else None
            # issue: keypoints에 스칼라값이 들어가는 문제
            if keypoints is not None and not keypoints.shape:
                keypoints = None
            # end of issue
            with owner._lock:
                # 현재 처리 결과가 최신인 경우에만. 즉, 이전의 timestamp(keypoints[0])보다
                # 지금 처리한 frame의 timestamp가 더 큰 경우에만 인정
                if owner._timestamp_and_keypoints[0] < timestamp:
                    owner._timestamp_and_keypoints = timestamp, keypoints


class FrameProcessor:
    def __init__(self, server_url, workers=4):
        self.__stream = BytesIO()
        self._condition = Condition()
        self._recent_frame = time(), None
        self.__pools = [KeypointsExtractor(server_url, self, f'keypoints_extractor-{i}') for i in range(workers)]
        self._lock = Lock()
        self._timestamp_and_keypoints = time(), None
    
    def write(self, buf: bytes):
        if buf.startswith(b'\xff\xd8') and self.__stream.tell() > 0:
            with self._condition:
                self._recent_frame = time(), self.__stream.getvalue()
                self._condition.notify()
            self.__stream.seek(0)
            self.__stream.truncate()
        self.__stream.write(buf)

    def flush(self):
        self.__stream.close()
        for p in self.__pools:
            p.terminated = True
            p.join()

    @property
    def timestamp_and_keypoints(self) -> Tuple[float, Optional[np.ndarray]]:
        with self._lock:
            replica = deepcopy(self._timestamp_and_keypoints)
        return replica

    @property
    def keypoints(self) -> Optional[np.ndarray]:
        with self._lock:
            replica = deepcopy(self._timestamp_and_keypoints[1])
        return replica
