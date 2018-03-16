import os
import subprocess

class Tensorboard:
    def __init__(self, path):
        self._path = os.path.abspath(path)
        self._tboard_proc = subprocess.Popen(["tensorboard", "--host=127.0.0.1", "--port=6006", "--logdir={0}".format(self._path)])


    def __del__(self):
        if self._tboard_proc is not None:
            self._tboard_proc.terminate()
