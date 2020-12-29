from collections import namedtuple
from flask import Flask, request
from time import sleep
import json

from controllers import MainController
from measurer import Mode


with open('config.json') as f:
    config = json.load(f, object_hook=lambda d: namedtuple('Config', d.keys())(*d.values()))

controller = MainController(config)
app = Flask(__name__)


@app.route('/application', methods=['POST'])
def application():
    content = request.get_json()
    command = content['command']

    msg = 'success'
    code = 200

    if command == 'exit':
        controller.close()
    elif command == 'reset':
        pass
    elif command == 'reserved':
        pass
    else:
        msg, code = 'undefined command.', 400
    
    return {'msg': msg}, code


@app.route('/', methods=['GET'])
def start_measuring():
    sleep(5)
    user_id = request.args.get('seq')
    if controller.bbm.state != Mode.Idle:
        return {'msg': 'Status is not in idle'}, 400
        
    result = controller.bbm.start_measuring(user_id)
    if not result:
        return {'msg': 'Keypoints does not detected.'}, 400
    
    return {'msg': 'Success'}
