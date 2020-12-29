import sys
from sys import platform
import os

import cv2
import numpy as np
from flask import Flask, request, jsonify

dir_path = r'D:/projects/openpose-1.5.0/build/examples/tutorial_api_python'
try:
    # Windows Import
    if platform == "win32":
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append(dir_path + '/../../python/openpose/Release')
        os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' + dir_path + '/../../bin;'
        import pyopenpose as op
    else:
        # Change these variables to point to the correct folder (Release/x64 etc.)
        sys.path.append('../../python')
        # If you run `make install` (default path is `/usr/local/python` for Ubuntu),
        # you can also access the OpenPose/python module from there.
        # This will install OpenPose and the python library at your desired installation path.
        # Ensure that this is in your python path in order to use it.
        # sys.path.append('/usr/local/python')
        from openpose import pyopenpose as op
except ImportError as e:
    print('Error: OpenPose library could not be found. '
          'Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
    raise e


params = {
    'model_folder': dir_path + '/../../../models/',
    'render_pose': 0,
    'number_people_max': 1
}

opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()

app = Flask(__name__)


@app.route('/skeleton', methods=['POST'])
def skeleton():
    if 'frame' not in request.files:
        return jsonify(code=404, error_msg='File not found.'), 404

    frame = request.files['frame']
    frame = np.frombuffer(frame.stream.read(), np.uint8)
    frame = cv2.imdecode(frame, cv2.IMREAD_UNCHANGED)

    datum.cvInputData = frame
    opWrapper.emplaceAndPop([datum])

    if datum.poseKeypoints.shape:
        return jsonify(code=0, keypoints=datum.poseKeypoints.tolist())
    else:
        return jsonify(code=1, keypoints=[])
