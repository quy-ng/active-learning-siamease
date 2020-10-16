from flask import Flask, render_template, request
import codecs
import os
import json
import uuid
import shlex
import subprocess


app = Flask(__name__)


@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        task_id = str(uuid.uuid4()).split('-')[-1]
        f = request.files['file']
        f.save(f'{task_id}.xlsx')
        # call(["python", "slave.py", "--data_path",  f"{task_id}.xlsx", "--task_id", "f{task_id}"])
        cmds = shlex.split(f"python slave.py --data_path {task_id}.xlsx --task_id {task_id}")
        subprocess.Popen(cmds, start_new_session=True)
        return {"task_id": task_id}


@app.route('/status', methods=['POST'])
def status():
    if request.method == 'POST':
        task_id = request.json['task_id']
        task_status = f'file_{task_id}_status.json'
        if os.path.isfile(task_status):
            existing_status = codecs.open(task_status, 'r', 'UTF-8').read()
            is_status_file_empty = len(existing_status.strip()) == 0
            if is_status_file_empty:
                os.remove(task_status)
                return {"status": '-1'}
            else:
                status = json.loads(existing_status)
                status['status'] = 1
                os.remove(task_status)
                return status
        else:
            return {"status": '-2'}


@app.route('/submit', methods=['POST'])
def submit():
    if request.method == 'POST':
        task_id = request.json['task_id']
        task_submit = f'file_{task_id}_submit.json'
        if os.path.isfile(task_submit):
            return "please retry"
        else:
            json.dump(request.json, codecs.open(task_submit, 'w', 'UTF-8'))
            return "ok"


if __name__ == '__main__':
    app.run(debug=True)
