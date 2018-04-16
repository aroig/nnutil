import os
import subprocess

class Tensorboard:
    def __init__(self, path):
        self._path = os.path.abspath(path)

    def __enter__(self):
        self._tboard_proc = subprocess.Popen([
            "tensorboard",
            "--port=6006",
            "--debugger_port=6007",
            "--logdir={0}".format(self._path)
        ])
        return self

    def __exit__(self, type, value, traceback):
        if self._tboard_proc is not None:
            self._tboard_proc.terminate()
